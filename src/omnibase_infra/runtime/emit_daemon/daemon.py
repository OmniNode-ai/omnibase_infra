# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Hook Event Daemon - Unix socket server for persistent Kafka event emission.

This module provides the EmitDaemon class that implements a Unix socket server
for receiving events from Claude Code hooks and publishing them to Kafka with
fire-and-forget semantics from the caller's perspective.

Architecture:
    ```
    +-----------------+     Unix Socket     +-------------+     Kafka     +-------+
    | Claude Code     | -----------------> | EmitDaemon  | ------------> | Kafka |
    | Hooks           |   JSON messages    | (this file) |   Events     | Topics|
    +-----------------+                    +-------------+               +-------+
                                                 |
                                                 v
                                           +------------+
                                           | Disk Spool |
                                           | (overflow) |
                                           +------------+
    ```

Features:
    - Unix domain socket server for low-latency local IPC
    - Bounded in-memory queue with disk spool overflow
    - Persistent Kafka connection with retry logic
    - Fire-and-forget semantics for callers
    - Graceful shutdown with queue drain
    - PID file management for process tracking
    - Health check endpoint for monitoring

Protocol:
    Request format: {"event_type": "prompt.submitted", "payload": {...}}\\n
    Response format: {"status": "queued"}\\n or {"status": "error", "reason": "..."}\\n

    Special commands:
    - {"command": "ping"}\\n -> {"status": "ok", "queue_size": N, "spool_size": M}\\n

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP

.. versionadded:: 0.2.6
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

from pydantic import ValidationError

from omnibase_core.errors import OnexError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.emit_daemon.config import ModelEmitDaemonConfig
from omnibase_infra.runtime.emit_daemon.event_registry import EventRegistry
from omnibase_infra.runtime.emit_daemon.model_daemon_request import (
    ModelDaemonEmitRequest,
    ModelDaemonPingRequest,
    parse_daemon_request,
)
from omnibase_infra.runtime.emit_daemon.model_daemon_response import (
    ModelDaemonErrorResponse,
    ModelDaemonPingResponse,
    ModelDaemonQueuedResponse,
)
from omnibase_infra.runtime.emit_daemon.queue import BoundedEventQueue, ModelQueuedEvent

logger = logging.getLogger(__name__)

# Poll interval for publisher loop when queue is empty (seconds)
PUBLISHER_POLL_INTERVAL_SECONDS: float = 0.1


class EmitDaemon:
    """Unix socket daemon for persistent Kafka event emission.

    Accepts events via Unix socket, queues them, and publishes to Kafka
    with fire-and-forget semantics from the caller's perspective.

    The daemon operates as follows:
        1. Listens on a Unix domain socket for incoming events
        2. Validates event payloads (type, size, required fields)
        3. Queues events in a bounded in-memory queue
        4. Background publisher loop dequeues and publishes to Kafka
        5. On publish failure, events are re-queued with exponential backoff
        6. On graceful shutdown, queue is drained to disk spool

    Attributes:
        config: Daemon configuration model
        queue: Bounded event queue with disk spool

    Example:
        ```python
        from omnibase_infra.runtime.emit_daemon import EmitDaemon, ModelEmitDaemonConfig

        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="kafka:9092",
            socket_path=Path("/tmp/emit.sock"),
        )

        daemon = EmitDaemon(config)
        await daemon.start()

        # Daemon runs until SIGTERM or SIGINT
        # Or call daemon.stop() programmatically
        ```
    """

    def __init__(
        self,
        config: ModelEmitDaemonConfig,
        event_bus: ProtocolEventBusLike | None = None,
    ) -> None:
        """Initialize daemon with config.

        If event_bus is None, creates EventBusKafka from config.

        Args:
            config: Daemon configuration model containing socket path,
                Kafka settings, queue limits, and timeout values.
            event_bus: Optional event bus for testing. If not provided,
                creates EventBusKafka from config.

        Example:
            ```python
            # Production usage
            config = ModelEmitDaemonConfig(kafka_bootstrap_servers="kafka:9092")
            daemon = EmitDaemon(config)

            # Testing with mock event bus
            mock_bus = MockEventBus()
            daemon = EmitDaemon(config, event_bus=mock_bus)
            ```
        """
        self._config = config
        self._event_bus: ProtocolEventBusLike | None = event_bus

        # Event registry for topic resolution and payload enrichment
        self._registry = EventRegistry(environment=config.environment)

        # Bounded event queue with disk spool overflow
        self._queue = BoundedEventQueue(
            max_memory_queue=config.max_memory_queue,
            max_spool_messages=config.max_spool_messages,
            max_spool_bytes=config.max_spool_bytes,
            spool_dir=config.spool_dir,
        )

        # Server state
        self._server: asyncio.Server | None = None
        self._publisher_task: asyncio.Task[None] | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Lock for shared state access
        self._lock = asyncio.Lock()

        logger.debug(
            "EmitDaemon initialized",
            extra={
                "socket_path": str(config.socket_path),
                "kafka_servers": config.kafka_bootstrap_servers,
                "max_memory_queue": config.max_memory_queue,
            },
        )

    @property
    def config(self) -> ModelEmitDaemonConfig:
        """Get the daemon configuration.

        Returns:
            The daemon configuration model.
        """
        return self._config

    @property
    def queue(self) -> BoundedEventQueue:
        """Get the event queue.

        Returns:
            The bounded event queue with disk spool.
        """
        return self._queue

    async def start(self) -> None:
        """Start the daemon.

        Performs the following startup sequence:
            1. Check for stale socket/PID and clean up
            2. Create PID file
            3. Load any spooled events from disk
            4. Initialize Kafka event bus
            5. Start Unix socket server
            6. Start publisher loop (background task)
            7. Setup signal handlers for graceful shutdown

        Raises:
            OSError: If socket creation fails
            RuntimeError: If another daemon is already running
        """
        async with self._lock:
            if self._running:
                logger.debug("EmitDaemon already running")
                return

            # Check and clean up stale socket/PID
            if self._check_stale_socket():
                self._cleanup_stale()
            elif self._config.pid_path.exists():
                # Another daemon is running
                pid = self._config.pid_path.read_text().strip()
                raise OnexError(
                    f"Another emit daemon is already running with PID {pid}"
                )

            # Create PID file
            self._write_pid_file()

            # Load any spooled events from previous runs
            spool_count = await self._queue.load_spool()
            if spool_count > 0:
                logger.info(f"Loaded {spool_count} events from spool")

            # Initialize Kafka event bus if not provided
            if self._event_bus is None:
                kafka_config = ModelKafkaEventBusConfig(
                    bootstrap_servers=self._config.kafka_bootstrap_servers,
                    environment=self._config.environment,
                    timeout_seconds=int(self._config.kafka_timeout_seconds),
                )
                self._event_bus = EventBusKafka(config=kafka_config)

            # Start the event bus (connects to Kafka)
            # NOTE: hasattr check required because event_bus can be a mock for testing
            # that may not implement start(). EventBusKafka always has start(), but
            # test doubles may omit it if they don't need explicit initialization.
            if hasattr(self._event_bus, "start"):
                await self._event_bus.start()  # type: ignore[union-attr]

            # Ensure parent directory exists for socket
            self._config.socket_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing socket file if present
            if self._config.socket_path.exists():
                self._config.socket_path.unlink()

            # Start Unix socket server
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self._config.socket_path),
            )

            # Set socket permissions (configurable, defaults to owner and group read/write)
            self._config.socket_path.chmod(self._config.socket_permissions)

            # Start publisher loop as background task
            self._publisher_task = asyncio.create_task(self._publisher_loop())

            # Setup signal handlers for graceful shutdown
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self._signal_handler)

            self._running = True
            self._shutdown_event.clear()

            logger.info(
                "EmitDaemon started",
                extra={
                    "socket_path": str(self._config.socket_path),
                    "pid": os.getpid(),
                },
            )

    async def stop(self) -> None:
        """Stop the daemon gracefully.

        Performs the following shutdown sequence:
            1. Stop accepting new connections
            2. Cancel publisher task
            3. Drain queue to spool (up to shutdown_drain_seconds)
            4. Close Kafka connection
            5. Remove socket and PID file

        This method is safe to call multiple times.
        """
        async with self._lock:
            if not self._running:
                logger.debug("EmitDaemon not running")
                return

            self._running = False
            self._shutdown_event.set()

            logger.info("EmitDaemon stopping...")

            # Remove signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)

            # Stop accepting new connections
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                self._server = None

            # Cancel publisher task
            if self._publisher_task is not None:
                self._publisher_task.cancel()
                try:
                    await self._publisher_task
                except asyncio.CancelledError:
                    pass
                self._publisher_task = None

            # Drain queue to spool with timeout
            if self._config.shutdown_drain_seconds > 0:
                try:
                    async with asyncio.timeout(self._config.shutdown_drain_seconds):
                        drained = await self._queue.drain_to_spool()
                        if drained > 0:
                            logger.info(f"Drained {drained} events to spool")
                except TimeoutError:
                    logger.warning(
                        "Shutdown drain timeout exceeded, some events may be lost"
                    )

            # Close Kafka connection
            # NOTE: hasattr check required because event_bus can be a mock for testing
            # that may not implement close(). EventBusKafka always has close(), but
            # test doubles may omit it if they don't need explicit cleanup.
            if self._event_bus is not None and hasattr(self._event_bus, "close"):
                await self._event_bus.close()  # type: ignore[union-attr]

            # Remove socket file
            if self._config.socket_path.exists():
                try:
                    self._config.socket_path.unlink()
                except OSError as e:
                    logger.warning(f"Failed to remove socket file: {e}")

            # Remove PID file
            self._remove_pid_file()

            logger.info("EmitDaemon stopped")

    async def run_until_shutdown(self) -> None:
        """Run the daemon until shutdown signal is received.

        Blocks until SIGTERM/SIGINT is received or stop() is called.
        Useful for running the daemon as a standalone process.

        Example:
            ```python
            daemon = EmitDaemon(config)
            await daemon.start()
            await daemon.run_until_shutdown()
            ```
        """
        await self._shutdown_event.wait()
        await self.stop()

    def _signal_handler(self) -> None:
        """Handle SIGTERM/SIGINT signals.

        Sets the shutdown event to trigger graceful shutdown.
        """
        logger.info("Received shutdown signal")
        self._shutdown_event.set()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection.

        Protocol: newline-delimited JSON
        Request: {"event_type": "...", "payload": {...}}
        Response: {"status": "queued"} or {"status": "error", "reason": "..."}

        Special commands:
        - {"command": "ping"} -> {"status": "ok", "queue_size": N, "spool_size": M}

        Args:
            reader: Async stream reader for the client connection
            writer: Async stream writer for the client connection
        """
        peer = "unix_client"
        logger.debug(f"Client connected: {peer}")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Read line with timeout
                    line = await asyncio.wait_for(
                        reader.readline(),
                        timeout=self._config.socket_timeout_seconds,
                    )
                except TimeoutError:
                    # Client timeout - close connection
                    logger.debug(f"Client timeout: {peer}")
                    break

                if not line:
                    # Client disconnected
                    break

                # Process the request
                response = await self._process_request(line)

                # Send response
                writer.write(response.encode("utf-8") + b"\n")
                await writer.drain()

        except ConnectionResetError:
            logger.debug(f"Client connection reset: {peer}")
        except Exception as e:
            logger.exception(f"Error handling client {peer}: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug(f"Client disconnected: {peer}")

    async def _process_request(self, line: bytes) -> str:
        """Process a single request line.

        Uses typed request models (ModelDaemonPingRequest, ModelDaemonEmitRequest)
        for compile-time type safety instead of dict[str, object] with isinstance checks.

        Args:
            line: Raw request line (JSON bytes with optional newline)

        Returns:
            JSON response string
        """
        try:
            # Parse JSON request
            raw_request = json.loads(line.decode("utf-8").strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return ModelDaemonErrorResponse(
                reason=f"Invalid JSON: {e}"
            ).model_dump_json()

        if not isinstance(raw_request, dict):
            return ModelDaemonErrorResponse(
                reason="Request must be a JSON object"
            ).model_dump_json()

        # Parse into typed request model
        try:
            request = parse_daemon_request(raw_request)
        except (ValueError, ValidationError) as e:
            return ModelDaemonErrorResponse(reason=str(e)).model_dump_json()

        # Dispatch based on request type
        if isinstance(request, ModelDaemonPingRequest):
            return await self._handle_ping(request)
        elif isinstance(request, ModelDaemonEmitRequest):
            return await self._handle_emit(request)
        else:
            # Should be unreachable due to exhaustive type check above
            return ModelDaemonErrorResponse(
                reason="Unknown request type"
            ).model_dump_json()

    async def _handle_ping(self, request: ModelDaemonPingRequest) -> str:
        """Handle ping command request.

        Args:
            request: Typed ping request model

        Returns:
            JSON response string with queue status
        """
        return ModelDaemonPingResponse(
            queue_size=self._queue.memory_size(),
            spool_size=self._queue.spool_size(),
        ).model_dump_json()

    async def _handle_emit(self, request: ModelDaemonEmitRequest) -> str:
        """Handle event emission request.

        Args:
            request: Typed emit request model with event_type and payload

        Returns:
            JSON response string (queued or error)
        """
        event_type = request.event_type

        # Normalize payload to dict (JsonType could be various types)
        raw_payload = request.payload
        if raw_payload is None:
            raw_payload = {}
        if not isinstance(raw_payload, dict):
            return ModelDaemonErrorResponse(
                reason="'payload' must be a JSON object"
            ).model_dump_json()

        # Cast to dict[str, object] after isinstance check for type safety
        payload: dict[str, object] = cast("dict[str, object]", raw_payload)

        # Check payload size
        payload_json = json.dumps(payload)
        if len(payload_json.encode("utf-8")) > self._config.max_payload_bytes:
            return ModelDaemonErrorResponse(
                reason=f"Payload exceeds maximum size of {self._config.max_payload_bytes} bytes"
            ).model_dump_json()

        # Validate event type is registered
        try:
            topic = self._registry.resolve_topic(event_type)
        except OnexError as e:
            return ModelDaemonErrorResponse(reason=str(e)).model_dump_json()

        # Validate payload has required fields
        try:
            self._registry.validate_payload(event_type, payload)
        except OnexError as e:
            return ModelDaemonErrorResponse(reason=str(e)).model_dump_json()

        # Extract correlation_id from payload if present
        correlation_id = payload.get("correlation_id")
        if not isinstance(correlation_id, str):
            correlation_id = None

        # Inject metadata into payload
        enriched_payload = self._registry.inject_metadata(
            event_type,
            payload,
            correlation_id=correlation_id,
        )

        # Get partition key
        partition_key = self._registry.get_partition_key(event_type, enriched_payload)

        # Create queued event
        event_id = str(uuid4())
        queued_event = ModelQueuedEvent(
            event_id=event_id,
            event_type=event_type,
            topic=topic,
            payload=enriched_payload,
            partition_key=partition_key,
            queued_at=datetime.now(UTC),
        )

        # Enqueue the event
        success = await self._queue.enqueue(queued_event)
        if success:
            logger.debug(
                f"Event queued: {event_id}",
                extra={
                    "event_type": event_type,
                    "topic": topic,
                },
            )
            return ModelDaemonQueuedResponse(event_id=event_id).model_dump_json()
        else:
            return ModelDaemonErrorResponse(
                reason="Failed to queue event (queue may be full)"
            ).model_dump_json()

    async def _publisher_loop(self) -> None:
        """Background task that dequeues and publishes events to Kafka.

        Runs continuously until stopped. On publish failure:
        - Increment retry_count
        - Re-queue with exponential backoff
        - After max_retry_attempts (from config), log error and drop event
        """
        logger.info("Publisher loop started")

        # NOTE: Using non-locking total_size() is intentional here.
        # While this creates a theoretical race condition during shutdown
        # (a concurrent enqueue could complete after the size check but before
        # the loop re-evaluates), it avoids lock contention in this hot loop.
        # For fire-and-forget semantics, this trade-off is acceptable - events
        # queued during the final shutdown window may be lost, which is
        # documented behavior (see shutdown_drain_seconds config and the
        # drain_to_spool() call in stop()). The queue's total_size_locked()
        # method exists for cases requiring accurate counts.
        while self._running or self._queue.total_size() > 0:
            try:
                # Dequeue next event
                event = await self._queue.dequeue()

                if event is None:
                    # Queue empty, wait briefly and check again
                    await asyncio.sleep(PUBLISHER_POLL_INTERVAL_SECONDS)
                    continue

                # Attempt to publish
                success = await self._publish_event(event)

                if not success:
                    # Increment retry count
                    event.retry_count += 1

                    if event.retry_count >= self._config.max_retry_attempts:
                        # Max retries exceeded - drop event
                        logger.error(
                            f"Dropping event {event.event_id} after {event.retry_count} retries",
                            extra={
                                "event_type": event.event_type,
                                "topic": event.topic,
                            },
                        )
                    else:
                        # Re-queue with backoff (capped to prevent excessive delays)
                        uncapped_backoff = self._config.backoff_base_seconds * (
                            2 ** (event.retry_count - 1)
                        )
                        backoff = min(
                            uncapped_backoff, self._config.max_backoff_seconds
                        )
                        logger.warning(
                            f"Publish failed for {event.event_id}, retry {event.retry_count}/{self._config.max_retry_attempts} in {backoff}s",
                            extra={
                                "event_type": event.event_type,
                                "topic": event.topic,
                            },
                        )

                        # Wait for backoff period
                        await asyncio.sleep(backoff)

                        # Re-enqueue with error handling
                        requeue_success = await self._queue.enqueue(event)
                        if not requeue_success:
                            logger.error(
                                f"Failed to re-enqueue event {event.event_id} after backoff, event lost",
                                extra={
                                    "event_type": event.event_type,
                                    "topic": event.topic,
                                    "retry_count": event.retry_count,
                                },
                            )

            except asyncio.CancelledError:
                logger.info("Publisher loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Unexpected error in publisher loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before continuing

        logger.info("Publisher loop stopped")

    async def _publish_event(self, event: ModelQueuedEvent) -> bool:
        """Publish a single event to Kafka.

        Args:
            event: The queued event to publish

        Returns:
            True if publish succeeded, False otherwise
        """
        if self._event_bus is None:
            logger.error("Event bus not initialized")
            return False

        try:
            # Prepare message key and value
            key = event.partition_key.encode("utf-8") if event.partition_key else None
            value = json.dumps(event.payload).encode("utf-8")

            # Extract correlation_id from enriched payload (injected by registry)
            # Type guard: payload is always a dict in practice (created in _handle_event)
            payload_correlation_id = (
                event.payload.get("correlation_id")
                if isinstance(event.payload, dict)
                else None
            )
            if isinstance(payload_correlation_id, str):
                try:
                    correlation_id = UUID(payload_correlation_id)
                except ValueError:
                    correlation_id = uuid4()
            else:
                correlation_id = uuid4()

            # Create event headers
            headers = ModelEventHeaders(
                source="emit-daemon",
                event_type=event.event_type,
                timestamp=event.queued_at,
                correlation_id=correlation_id,
            )

            # Publish to Kafka
            # NOTE: headers parameter is Kafka-specific, not in minimal protocol.
            # When _event_bus is None, we create EventBusKafka which supports headers.
            # For testing mocks, they can accept **kwargs or ignore extra params.
            await self._event_bus.publish(  # type: ignore[call-arg]
                topic=event.topic,
                key=key,
                value=value,
                headers=headers,
            )

            logger.debug(
                f"Published event {event.event_id}",
                extra={
                    "event_type": event.event_type,
                    "topic": event.topic,
                },
            )
            return True

        except Exception as e:
            logger.warning(
                f"Failed to publish event {event.event_id}: {e}",
                extra={
                    "event_type": event.event_type,
                    "topic": event.topic,
                    "error": str(e),
                },
            )
            return False

    def _write_pid_file(self) -> None:
        """Write current PID to pid_path.

        Creates parent directories if needed.
        """
        try:
            self._config.pid_path.parent.mkdir(parents=True, exist_ok=True)
            self._config.pid_path.write_text(str(os.getpid()))
            logger.debug(f"PID file created: {self._config.pid_path}")
        except OSError as e:
            logger.warning(f"Failed to write PID file: {e}")

    def _remove_pid_file(self) -> None:
        """Remove PID file if it exists."""
        try:
            if self._config.pid_path.exists():
                self._config.pid_path.unlink()
                logger.debug(f"PID file removed: {self._config.pid_path}")
        except OSError as e:
            logger.warning(f"Failed to remove PID file: {e}")

    def _check_stale_socket(self) -> bool:
        """Check if socket/PID are stale (process not running).

        A socket/PID is considered stale if:
        - PID file exists but the process is not running
        - Socket file exists but no PID file exists

        Returns:
            True if stale (safe to clean up), False if daemon is running.
        """
        # Check if PID file exists
        if not self._config.pid_path.exists():
            # No PID file - socket is stale if it exists
            return self._config.socket_path.exists()

        # Read PID from file
        try:
            pid_str = self._config.pid_path.read_text().strip()
            pid = int(pid_str)
        except (OSError, ValueError):
            # Can't read PID file - treat as stale
            return True

        # Check if process is running
        try:
            # Sending signal 0 checks if process exists without killing it
            os.kill(pid, 0)
            # Process is running - not stale
            return False
        except ProcessLookupError:
            # Process not running - stale
            return True
        except PermissionError:
            # Process exists but we can't signal it - assume not stale
            return False

    def _cleanup_stale(self) -> None:
        """Remove stale socket and PID files."""
        # Remove socket file
        if self._config.socket_path.exists():
            try:
                self._config.socket_path.unlink()
                logger.info(f"Removed stale socket: {self._config.socket_path}")
            except OSError as e:
                logger.warning(f"Failed to remove stale socket: {e}")

        # Remove PID file
        if self._config.pid_path.exists():
            try:
                self._config.pid_path.unlink()
                logger.info(f"Removed stale PID file: {self._config.pid_path}")
            except OSError as e:
                logger.warning(f"Failed to remove stale PID file: {e}")


__all__: list[str] = ["EmitDaemon"]

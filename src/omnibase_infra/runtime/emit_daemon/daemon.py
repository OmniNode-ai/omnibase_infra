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
from uuid import UUID, uuid4

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.emit_daemon.config import ModelEmitDaemonConfig
from omnibase_infra.runtime.emit_daemon.event_registry import EventRegistry
from omnibase_infra.runtime.emit_daemon.queue import BoundedEventQueue, ModelQueuedEvent

logger = logging.getLogger(__name__)


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
        event_bus: EventBusKafka | None = None,
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
        self._event_bus: EventBusKafka | None = event_bus

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
                raise RuntimeError(
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
                    environment="dev",
                    timeout_seconds=int(self._config.kafka_timeout_seconds),
                )
                self._event_bus = EventBusKafka(config=kafka_config)

            # Start the event bus (connects to Kafka)
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

            # Set socket permissions (readable/writable by owner and group)
            self._config.socket_path.chmod(0o660)

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

        Args:
            line: Raw request line (JSON bytes with optional newline)

        Returns:
            JSON response string
        """
        try:
            # Parse JSON request
            request = json.loads(line.decode("utf-8").strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return json.dumps({"status": "error", "reason": f"Invalid JSON: {e}"})

        if not isinstance(request, dict):
            return json.dumps(
                {"status": "error", "reason": "Request must be a JSON object"}
            )

        # Handle special commands
        if "command" in request:
            return await self._handle_command(request)

        # Handle event submission
        return await self._handle_event(request)

    async def _handle_command(self, request: dict[str, object]) -> str:
        """Handle special command requests.

        Args:
            request: Parsed command request

        Returns:
            JSON response string
        """
        command = request.get("command")

        if command == "ping":
            return json.dumps(
                {
                    "status": "ok",
                    "queue_size": self._queue.memory_size(),
                    "spool_size": self._queue.spool_size(),
                }
            )

        return json.dumps({"status": "error", "reason": f"Unknown command: {command}"})

    async def _handle_event(self, request: dict[str, object]) -> str:
        """Handle event submission requests.

        Args:
            request: Parsed event request with event_type and payload

        Returns:
            JSON response string
        """
        # Validate required fields
        event_type = request.get("event_type")
        if not event_type or not isinstance(event_type, str):
            return json.dumps(
                {
                    "status": "error",
                    "reason": "Missing or invalid 'event_type' field",
                }
            )

        payload = request.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return json.dumps(
                {
                    "status": "error",
                    "reason": "'payload' must be a JSON object",
                }
            )

        # Check payload size
        payload_json = json.dumps(payload)
        if len(payload_json.encode("utf-8")) > self._config.max_payload_bytes:
            return json.dumps(
                {
                    "status": "error",
                    "reason": f"Payload exceeds maximum size of {self._config.max_payload_bytes} bytes",
                }
            )

        # Validate event type is registered
        try:
            topic = self._registry.resolve_topic(event_type)
        except ValueError as e:
            return json.dumps({"status": "error", "reason": str(e)})

        # Validate payload has required fields
        try:
            self._registry.validate_payload(event_type, payload)
        except ValueError as e:
            return json.dumps({"status": "error", "reason": str(e)})

        # Extract correlation_id from payload if present
        correlation_id = payload.get("correlation_id")
        if isinstance(correlation_id, str):
            pass  # Use as-is
        else:
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
            return json.dumps({"status": "queued", "event_id": event_id})
        else:
            return json.dumps(
                {
                    "status": "error",
                    "reason": "Failed to queue event (queue may be full)",
                }
            )

    async def _publisher_loop(self) -> None:
        """Background task that dequeues and publishes events to Kafka.

        Runs continuously until stopped. On publish failure:
        - Increment retry_count
        - Re-queue with exponential backoff
        - After max_retry_attempts (from config), log error and drop event
        """
        logger.info("Publisher loop started")

        while self._running or self._queue.total_size() > 0:
            try:
                # Dequeue next event
                event = await self._queue.dequeue()

                if event is None:
                    # Queue empty, wait briefly and check again
                    await asyncio.sleep(0.1)
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
                        # Re-queue with backoff
                        backoff = self._config.backoff_base_seconds * (
                            2 ** (event.retry_count - 1)
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

                        # Re-enqueue
                        await self._queue.enqueue(event)

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
            payload_correlation_id = event.payload.get("correlation_id")
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
            await self._event_bus.publish(
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

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Emit Daemon Client - Python client for emitting events via the daemon.

This module provides the EmitClient class for communicating with the EmitDaemon
via Unix socket. It offers fire-and-forget semantics where the client returns
as soon as the daemon acknowledges the event has been queued.

Features:
    - Async and sync interfaces for flexibility
    - Automatic connection management
    - Timeout handling with configurable limits
    - Health check (ping) support
    - Graceful degradation with fallback callback

Protocol:
    Request: {"event_type": "...", "payload": {...}}\\n
    Response: {"status": "queued", "event_id": "..."}\\n or {"status": "error", "reason": "..."}\\n
    Ping: {"command": "ping"}\\n -> {"status": "ok", "queue_size": N, "spool_size": M}\\n

Example:
    ```python
    from omnibase_infra.runtime.emit_daemon.client import EmitClient

    # Async usage
    async with EmitClient() as client:
        event_id = await client.emit("prompt.submitted", {"prompt_id": "abc123"})

    # Sync usage (for scripts)
    client = EmitClient()
    event_id = client.emit_sync("prompt.submitted", {"prompt_id": "abc123"})

    # Convenience function
    event_id = await emit_event("prompt.submitted", {"prompt_id": "abc123"})
    ```

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP

.. versionadded:: 0.2.6
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from types import TracebackType
from typing import TypeVar

from pydantic import ValidationError

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.errors import OnexError
from omnibase_core.types import JsonType
from omnibase_infra.runtime.emit_daemon.model_daemon_request import (
    ModelDaemonEmitRequest,
    ModelDaemonPingRequest,
)
from omnibase_infra.runtime.emit_daemon.model_daemon_response import (
    ModelDaemonErrorResponse,
    ModelDaemonPingResponse,
    ModelDaemonQueuedResponse,
    parse_daemon_response,
)

logger = logging.getLogger(__name__)

# Type variable for generic async runner
_T = TypeVar("_T")


class EmitClientError(OnexError):
    """Error communicating with emit daemon.

    Raised when the client cannot connect to the daemon, the daemon rejects
    the event, or a timeout occurs during communication.

    Inherits from OnexError for consistent error handling across the ONEX platform.

    Attributes:
        reason: Optional detailed reason for the error (from daemon response).

    Example:
        ```python
        try:
            await client.emit("event.type", {"data": "value"})
        except EmitClientError as e:
            print(f"Failed to emit: {e}")
            if e.reason:
                print(f"Reason: {e.reason}")
        ```
    """

    def __init__(
        self,
        message: str,
        reason: str | None = None,
        error_code: EnumCoreErrorCode = EnumCoreErrorCode.OPERATION_FAILED,
    ) -> None:
        """Initialize the error with a message and optional reason.

        Args:
            message: Human-readable error message
            reason: Optional detailed reason from daemon response
            error_code: Error code from EnumCoreErrorCode, defaults to OPERATION_FAILED
        """
        super().__init__(message=message, error_code=error_code)
        self.reason = reason


class EmitClient:
    """Client for emitting events via the emit daemon.

    Connects to the daemon's Unix socket and sends events with fire-and-forget
    semantics. The client returns as soon as the daemon acknowledges the event
    has been queued for Kafka publishing.

    The client supports both async context manager usage for connection pooling
    and standalone usage where each operation creates a new connection.

    Attributes:
        socket_path: Path to the daemon's Unix socket
        timeout: Timeout in seconds for socket operations

    Example:
        ```python
        # Context manager usage (recommended for multiple operations)
        async with EmitClient() as client:
            await client.emit("event.one", {"data": "1"})
            await client.emit("event.two", {"data": "2"})

        # Standalone usage (creates new connection per operation)
        client = EmitClient()
        await client.emit("event.single", {"data": "value"})

        # Custom socket path and timeout
        client = EmitClient(
            socket_path=Path("/custom/emit.sock"),
            timeout=10.0,
        )
        ```
    """

    # Default socket path matching daemon config default
    # NOTE: /tmp is standard for Unix domain sockets - not a security issue
    DEFAULT_SOCKET_PATH: Path = Path("/tmp/omniclaude-emit.sock")  # noqa: S108

    # Default timeout in seconds
    DEFAULT_TIMEOUT: float = 5.0

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize client with socket path and timeout.

        Args:
            socket_path: Path to the daemon's Unix socket. Defaults to
                /tmp/omniclaude-emit.sock to match daemon default.
            timeout: Timeout in seconds for socket operations. Defaults to 5.0.

        Example:
            ```python
            # Use defaults
            client = EmitClient()

            # Custom configuration
            client = EmitClient(
                socket_path="/var/run/emit.sock",
                timeout=10.0,
            )
            ```
        """
        self._socket_path = (
            Path(socket_path) if isinstance(socket_path, str) else socket_path
        )
        self._timeout = timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

        logger.debug(
            "EmitClient initialized",
            extra={
                "socket_path": str(self._socket_path),
                "timeout": self._timeout,
            },
        )

    @property
    def socket_path(self) -> Path:
        """Get the socket path.

        Returns:
            Path to the daemon's Unix socket.
        """
        return self._socket_path

    @property
    def timeout(self) -> float:
        """Get the timeout value.

        Returns:
            Timeout in seconds for socket operations.
        """
        return self._timeout

    async def __aenter__(self) -> EmitClient:
        """Enter async context manager, establishing connection.

        Returns:
            Self for use in async with statement.

        Raises:
            EmitClientError: If connection to daemon fails.
        """
        await self._connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing connection.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        await self._disconnect()

    async def _connect_unlocked(self) -> None:
        """Establish connection to daemon socket (without lock).

        Internal method - caller must hold self._lock.

        Raises:
            EmitClientError: If connection fails (daemon not running, permission denied, etc.)
        """
        if self._writer is not None:
            return  # Already connected

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._socket_path)),
                timeout=self._timeout,
            )
            logger.debug(f"Connected to emit daemon at {self._socket_path}")
        except FileNotFoundError as e:
            raise EmitClientError(
                f"Emit daemon socket not found at {self._socket_path}. "
                "Is the daemon running?",
                reason="socket_not_found",
                error_code=EnumCoreErrorCode.RESOURCE_NOT_FOUND,
            ) from e
        except PermissionError as e:
            raise EmitClientError(
                f"Permission denied accessing emit daemon socket at {self._socket_path}",
                reason="permission_denied",
                error_code=EnumCoreErrorCode.PERMISSION_DENIED,
            ) from e
        except ConnectionRefusedError as e:
            raise EmitClientError(
                f"Connection refused to emit daemon at {self._socket_path}. "
                "Is the daemon running?",
                reason="connection_refused",
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE,
            ) from e
        except TimeoutError as e:
            raise EmitClientError(
                f"Timeout connecting to emit daemon at {self._socket_path}",
                reason="connection_timeout",
                error_code=EnumCoreErrorCode.TIMEOUT_ERROR,
            ) from e
        except OSError as e:
            raise EmitClientError(
                f"Failed to connect to emit daemon: {e}",
                reason="os_error",
                error_code=EnumCoreErrorCode.NETWORK_ERROR,
            ) from e

    async def _connect(self) -> None:
        """Establish connection to daemon socket.

        Internal method called by context manager or lazily on first operation.

        Raises:
            EmitClientError: If connection fails (daemon not running, permission denied, etc.)
        """
        async with self._lock:
            await self._connect_unlocked()

    async def _disconnect_unlocked(self) -> None:
        """Close connection to daemon socket (without lock).

        Internal method - caller must hold self._lock.
        Safe to call multiple times.
        """
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error during disconnect cleanup: {e}")
            finally:
                self._writer = None
                self._reader = None
                logger.debug("Disconnected from emit daemon")

    async def _disconnect(self) -> None:
        """Close connection to daemon socket.

        Internal method called by context manager or cleanup operations.
        Safe to call multiple times.
        """
        async with self._lock:
            await self._disconnect_unlocked()

    async def _send_request(self, request: JsonType) -> dict[str, object]:
        """Send a request and receive response.

        Internal method for protocol communication. Acquires lock to ensure
        the write-then-read sequence is atomic, preventing response mixing
        when multiple coroutines call emit() concurrently.

        Args:
            request: Request dict to send (will be JSON-encoded)

        Returns:
            Response dict from daemon

        Raises:
            EmitClientError: If communication fails
        """
        async with self._lock:
            # Ensure we're connected (use unlocked variant since we hold the lock)
            if self._writer is None or self._reader is None:
                await self._connect_unlocked()

            # Type guard - we just connected, so these should be set
            if self._writer is None or self._reader is None:
                raise EmitClientError(
                    "Failed to establish connection", reason="no_connection"
                )

            try:
                # Send request
                request_json = json.dumps(request) + "\n"
                self._writer.write(request_json.encode("utf-8"))
                await self._writer.drain()

                # Receive response with timeout
                response_line = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=self._timeout,
                )

                if not response_line:
                    # Connection closed by daemon
                    await self._disconnect_unlocked()
                    raise EmitClientError(
                        "Connection closed by emit daemon",
                        reason="connection_closed",
                    )

                # Parse response
                response = json.loads(response_line.decode("utf-8").strip())
                if not isinstance(response, dict):
                    raise EmitClientError(
                        "Invalid response from daemon: expected JSON object",
                        reason="invalid_response",
                    )
                return response

            except TimeoutError as e:
                await self._disconnect_unlocked()
                raise EmitClientError(
                    f"Timeout waiting for daemon response (timeout={self._timeout}s)",
                    reason="response_timeout",
                    error_code=EnumCoreErrorCode.TIMEOUT_ERROR,
                ) from e
            except json.JSONDecodeError as e:
                await self._disconnect_unlocked()
                raise EmitClientError(
                    f"Invalid JSON response from daemon: {e}",
                    reason="invalid_json",
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                ) from e
            except ConnectionResetError as e:
                await self._disconnect_unlocked()
                raise EmitClientError(
                    "Connection reset by emit daemon",
                    reason="connection_reset",
                    error_code=EnumCoreErrorCode.NETWORK_ERROR,
                ) from e
            except BrokenPipeError as e:
                await self._disconnect_unlocked()
                raise EmitClientError(
                    "Broken pipe to emit daemon",
                    reason="broken_pipe",
                    error_code=EnumCoreErrorCode.NETWORK_ERROR,
                ) from e

    async def emit(
        self,
        event_type: str,
        payload: JsonType,
    ) -> str:
        """Emit an event via the daemon.

        Sends an event to the daemon for asynchronous publishing to Kafka.
        Returns as soon as the daemon acknowledges the event has been queued.

        Args:
            event_type: Semantic event type (e.g., "prompt.submitted",
                "tool.invoked"). Must be registered with the daemon's
                EventRegistry.
            payload: Event payload dict. Must contain required fields for
                the event type as defined in EventRegistry.

        Returns:
            Event ID assigned by the daemon (UUID string). This can be used
            for tracking/debugging but is not needed for normal operation.

        Raises:
            EmitClientError: If daemon is unavailable, rejects the event,
                or a timeout occurs.

        Example:
            ```python
            event_id = await client.emit(
                "prompt.submitted",
                {
                    "prompt_id": "abc123",
                    "session_id": "sess-456",
                    "prompt_text": "Hello, Claude!",
                },
            )
            print(f"Event queued with ID: {event_id}")
            ```
        """
        # Build typed request
        request = ModelDaemonEmitRequest(event_type=event_type, payload=payload)
        raw_response = await self._send_request(request.model_dump())

        # Parse typed response
        try:
            response = parse_daemon_response(raw_response)
        except (ValueError, ValidationError) as e:
            raise EmitClientError(
                f"Invalid daemon response: {e}",
                reason="invalid_response",
            ) from e

        # Handle response by type
        if isinstance(response, ModelDaemonQueuedResponse):
            logger.debug(
                f"Event emitted: {response.event_id}",
                extra={
                    "event_type": event_type,
                    "event_id": response.event_id,
                },
            )
            return response.event_id
        elif isinstance(response, ModelDaemonErrorResponse):
            raise EmitClientError(
                f"Daemon rejected event: {response.reason}",
                reason=response.reason,
            )
        elif isinstance(response, ModelDaemonPingResponse):
            # Unexpected ping response to emit request
            raise EmitClientError(
                "Unexpected ping response to emit request",
                reason="unexpected_response_type",
            )
        else:
            # Should be unreachable
            raise EmitClientError(
                "Unexpected daemon response type",
                reason="unexpected_status",
            )

    async def ping(self) -> ModelDaemonPingResponse:
        """Health check the daemon.

        Sends a ping command to the daemon to verify it is running and
        get current queue status.

        Returns:
            ModelDaemonPingResponse with:
            - status: "ok" (always for successful ping)
            - queue_size: Number of events in memory queue
            - spool_size: Number of events spooled to disk

        Raises:
            EmitClientError: If daemon is unavailable or returns error.

        Example:
            ```python
            status = await client.ping()
            print(f"Queue size: {status.queue_size}")
            print(f"Spool size: {status.spool_size}")
            ```
        """
        # Build typed request
        request = ModelDaemonPingRequest()
        raw_response = await self._send_request(request.model_dump())

        # Parse typed response
        try:
            response = parse_daemon_response(raw_response)
        except (ValueError, ValidationError) as e:
            raise EmitClientError(
                f"Invalid daemon response: {e}",
                reason="invalid_response",
            ) from e

        # Handle response by type
        if isinstance(response, ModelDaemonPingResponse):
            logger.debug(
                "Daemon ping successful",
                extra={
                    "queue_size": response.queue_size,
                    "spool_size": response.spool_size,
                },
            )
            return response
        elif isinstance(response, ModelDaemonErrorResponse):
            raise EmitClientError(
                f"Daemon ping error: {response.reason}",
                reason=response.reason,
            )
        elif isinstance(response, ModelDaemonQueuedResponse):
            # Unexpected queued response to ping request
            raise EmitClientError(
                "Unexpected queued response to ping request",
                reason="unexpected_response_type",
            )
        else:
            # Should be unreachable
            raise EmitClientError(
                "Unexpected daemon ping response type",
                reason="unexpected_status",
            )

    async def is_daemon_running(self) -> bool:
        """Check if daemon is running and responsive.

        Attempts to ping the daemon and returns True if successful.
        Unlike ping(), this method does not raise exceptions - it simply
        returns False if the daemon is unavailable.

        Returns:
            True if daemon responds to ping, False otherwise.

        Example:
            ```python
            if await client.is_daemon_running():
                await client.emit("event.type", {"data": "value"})
            else:
                # Fall back to direct Kafka publish
                await direct_publish("event.type", {"data": "value"})
            ```
        """
        try:
            await self.ping()
            return True
        except EmitClientError:
            return False
        except Exception:
            return False

    def emit_sync(
        self,
        event_type: str,
        payload: JsonType,
    ) -> str:
        """Synchronous wrapper for emit().

        Creates an event loop if needed, calls emit(), and returns the result.
        Useful for shell scripts and non-async code.

        Note: This method creates a new connection for each call. For multiple
        operations, consider using async code with the context manager.

        Args:
            event_type: Semantic event type (e.g., "prompt.submitted")
            payload: Event payload dict

        Returns:
            Event ID assigned by the daemon

        Raises:
            EmitClientError: If daemon is unavailable or rejects the event

        Example:
            ```python
            # In a non-async context (shell script, simple script)
            client = EmitClient()
            event_id = client.emit_sync("prompt.submitted", {"prompt_id": "abc"})
            ```
        """
        return self._run_async(self._emit_and_disconnect(event_type, payload))

    def ping_sync(self) -> ModelDaemonPingResponse:
        """Synchronous wrapper for ping().

        Creates an event loop if needed, calls ping(), and returns the result.
        Useful for shell scripts and health checks.

        Returns:
            ModelDaemonPingResponse with status, queue_size, spool_size

        Raises:
            EmitClientError: If daemon is unavailable

        Example:
            ```python
            status = EmitClient().ping_sync()
            print(f"Daemon healthy, queue size: {status.queue_size}")
            ```
        """
        return self._run_async(self._ping_and_disconnect())

    def is_daemon_running_sync(self) -> bool:
        """Synchronous wrapper for is_daemon_running().

        Returns:
            True if daemon responds to ping, False otherwise.
        """
        return self._run_async(self.is_daemon_running())

    async def _emit_and_disconnect(
        self,
        event_type: str,
        payload: JsonType,
    ) -> str:
        """Emit event and disconnect (for sync wrapper).

        Args:
            event_type: Event type to emit
            payload: Event payload

        Returns:
            Event ID from daemon
        """
        try:
            return await self.emit(event_type, payload)
        finally:
            await self._disconnect()

    async def _ping_and_disconnect(self) -> ModelDaemonPingResponse:
        """Ping daemon and disconnect (for sync wrapper).

        Returns:
            Typed ping response from daemon
        """
        try:
            return await self.ping()
        finally:
            await self._disconnect()

    def _run_async(self, coro: Coroutine[object, object, _T]) -> _T:
        """Run an async coroutine in a sync context.

        Handles event loop creation for sync wrappers.

        Args:
            coro: Coroutine to run

        Returns:
            Result from the coroutine
        """
        try:
            # Check if we're in an existing event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create a new one
            loop = None

        if loop is not None:
            # We're in an async context - use run_until_complete is not allowed
            # Create a new event loop in a thread would be complex
            # Instead, raise an error suggesting async usage
            raise RuntimeError(
                "Cannot use sync methods from an async context. "
                "Use the async emit() or ping() methods instead."
            )

        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)


# Module-level convenience functions


async def emit_event(
    event_type: str,
    payload: JsonType,
    socket_path: Path | str = EmitClient.DEFAULT_SOCKET_PATH,
    timeout: float = EmitClient.DEFAULT_TIMEOUT,
) -> str:
    """Convenience function to emit a single event.

    Creates a client, emits the event, and closes the connection.
    For multiple events, use EmitClient as a context manager instead.

    Args:
        event_type: Semantic event type (e.g., "prompt.submitted")
        payload: Event payload dict
        socket_path: Path to daemon socket (defaults to /tmp/omniclaude-emit.sock)
        timeout: Timeout in seconds for socket operations (defaults to 5.0)

    Returns:
        Event ID assigned by the daemon

    Raises:
        EmitClientError: If daemon is unavailable or rejects the event

    Example:
        ```python
        event_id = await emit_event(
            "prompt.submitted",
            {"prompt_id": "abc123", "prompt_text": "Hello!"},
        )
        ```
    """
    async with EmitClient(socket_path=socket_path, timeout=timeout) as client:
        return await client.emit(event_type, payload)


async def emit_event_with_fallback(
    event_type: str,
    payload: JsonType,
    socket_path: Path | str = EmitClient.DEFAULT_SOCKET_PATH,
    timeout: float = EmitClient.DEFAULT_TIMEOUT,
    fallback: Callable[[str, JsonType], Awaitable[str]] | None = None,
) -> str:
    """Emit event via daemon, falling back to callback if daemon unavailable.

    Use this when you want graceful degradation to direct Kafka publish
    (or other fallback mechanism) when the daemon is not running.

    Args:
        event_type: Semantic event type (e.g., "prompt.submitted")
        payload: Event payload dict
        socket_path: Path to daemon socket (defaults to /tmp/omniclaude-emit.sock)
        timeout: Timeout in seconds for socket operations (defaults to 5.0)
        fallback: Optional async callback to invoke if daemon is unavailable.
            Receives (event_type, payload) and should return event_id string.
            If None and daemon is unavailable, raises EmitClientError.

    Returns:
        Event ID from daemon or fallback

    Raises:
        EmitClientError: If daemon unavailable and no fallback provided,
            or if fallback raises an exception.

    Example:
        ```python
        async def direct_kafka_publish(event_type: str, payload: dict) -> str:
            # Direct Kafka publish implementation
            return str(uuid4())

        event_id = await emit_event_with_fallback(
            "prompt.submitted",
            {"prompt_id": "abc123"},
            fallback=direct_kafka_publish,
        )
        ```
    """
    client = EmitClient(socket_path=socket_path, timeout=timeout)

    try:
        async with client:
            return await client.emit(event_type, payload)
    except EmitClientError as e:
        # Check if this is a connection error (daemon not running)
        connection_errors = {
            "socket_not_found",
            "connection_refused",
            "connection_timeout",
            "permission_denied",
            "os_error",
        }

        if e.reason in connection_errors and fallback is not None:
            logger.info(
                f"Daemon unavailable ({e.reason}), using fallback",
                extra={"event_type": event_type},
            )
            return await fallback(event_type, payload)

        # Re-raise if not a connection error or no fallback
        raise


__all__: list[str] = [
    "EmitClient",
    "EmitClientError",
    "emit_event",
    "emit_event_with_fallback",
]

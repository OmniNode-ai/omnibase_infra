# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S108  # /tmp paths are standard for Unix socket testing
"""Unit tests for EmitClient - Python client for the emit daemon.

This test suite validates:
- EmitClient.emit() success and error handling
- EmitClient.ping() health check functionality
- EmitClient.is_daemon_running() non-throwing availability check
- Socket error handling (not found, connection refused, timeout)
- Sync wrappers (emit_sync, ping_sync)
- Module-level convenience functions (emit_event, emit_event_with_fallback)
- EmitClientError exception attributes

Test Organization:
    - TestEmitClientEmit: emit() method success and error cases
    - TestEmitClientPing: ping() method and health checks
    - TestEmitClientIsDaemonRunning: is_daemon_running() boolean checks
    - TestEmitClientSocketErrors: Connection error scenarios
    - TestEmitClientSyncWrappers: Synchronous wrapper methods
    - TestEmitEventHelper: emit_event() convenience function
    - TestEmitEventWithFallback: emit_event_with_fallback() with fallback logic
    - TestEmitClientError: Exception class structure

Mocking Strategy:
    - Mock asyncio.open_unix_connection to return fake reader/writer
    - Mock reader.readline() to return canned JSON responses
    - Test error conditions by making mock raise exceptions

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP

.. versionadded:: 0.2.6
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.emit_daemon.client import (
    EmitClient,
    EmitClientError,
    emit_event,
    emit_event_with_fallback,
)


def create_mock_reader_writer(
    response: dict[str, object],
) -> tuple[AsyncMock, MagicMock]:
    """Create mock reader and writer for socket tests.

    Args:
        response: Dict to return as JSON from reader.readline()

    Returns:
        Tuple of (mock_reader, mock_writer)
    """
    mock_reader = AsyncMock()
    mock_reader.readline = AsyncMock(
        return_value=(json.dumps(response) + "\n").encode("utf-8")
    )

    mock_writer = MagicMock()
    mock_writer.write = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    return mock_reader, mock_writer


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmitClientEmit:
    """Test EmitClient.emit() method success and error cases."""

    async def test_emit_success_returns_event_id(self) -> None:
        """Test that emit() returns event_id on successful queue."""
        expected_event_id = "550e8400-e29b-41d4-a716-446655440000"
        response = {"status": "queued", "event_id": expected_event_id}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            event_id = await client.emit("test.event", {"key": "value"})

            assert event_id == expected_event_id
            mock_writer.write.assert_called_once()
            mock_writer.drain.assert_awaited_once()
            mock_reader.readline.assert_awaited_once()

    async def test_emit_sends_correct_request_format(self) -> None:
        """Test that emit() sends correctly formatted JSON request."""
        response = {"status": "queued", "event_id": "test-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            await client.emit("prompt.submitted", {"prompt_id": "abc123"})

            # Verify the request format
            call_args = mock_writer.write.call_args[0][0]
            request = json.loads(call_args.decode("utf-8").strip())

            assert request["event_type"] == "prompt.submitted"
            assert request["payload"] == {"prompt_id": "abc123"}

    async def test_emit_error_response_raises_emit_client_error(self) -> None:
        """Test that daemon error response raises EmitClientError with reason."""
        error_reason = "payload too large"
        response = {"status": "error", "reason": error_reason}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"data": "x" * 10000})

            assert error_reason in str(exc_info.value)
            assert exc_info.value.reason == error_reason

    async def test_emit_unexpected_status_raises_emit_client_error(self) -> None:
        """Test that unexpected status in response raises EmitClientError."""
        response = {"status": "unknown_status"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert "unexpected" in str(exc_info.value).lower()
            assert exc_info.value.reason == "unexpected_status"

    async def test_emit_invalid_event_id_raises_emit_client_error(self) -> None:
        """Test that response with invalid event_id raises EmitClientError."""
        response = {"status": "queued", "event_id": 12345}  # Not a string
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert exc_info.value.reason == "invalid_event_id"

    async def test_emit_context_manager_usage(self) -> None:
        """Test emit() works with async context manager."""
        response = {"status": "queued", "event_id": "ctx-mgr-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            async with EmitClient(socket_path="/tmp/test.sock") as client:
                event_id = await client.emit("test.event", {"key": "value"})
                assert event_id == "ctx-mgr-id"

            # Writer should be closed on exit
            mock_writer.close.assert_called_once()
            mock_writer.wait_closed.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmitClientPing:
    """Test EmitClient.ping() health check functionality."""

    async def test_ping_success_returns_status_dict(self) -> None:
        """Test that ping() returns dict with status on success."""
        response = {"status": "ok", "queue_size": 5, "spool_size": 10}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            result = await client.ping()

            assert result["status"] == "ok"
            assert result["queue_size"] == 5
            assert result["spool_size"] == 10

    async def test_ping_sends_correct_command(self) -> None:
        """Test that ping() sends correct command format."""
        response = {"status": "ok", "queue_size": 0, "spool_size": 0}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            await client.ping()

            call_args = mock_writer.write.call_args[0][0]
            request = json.loads(call_args.decode("utf-8").strip())

            assert request == {"command": "ping"}

    async def test_ping_error_response_raises_emit_client_error(self) -> None:
        """Test that ping error response raises EmitClientError."""
        response = {"status": "error", "reason": "internal_error"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.ping()

            assert exc_info.value.reason == "internal_error"


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmitClientIsDaemonRunning:
    """Test EmitClient.is_daemon_running() non-throwing availability check."""

    async def test_is_daemon_running_returns_true_when_ping_succeeds(self) -> None:
        """Test is_daemon_running() returns True when daemon responds."""
        response = {"status": "ok", "queue_size": 0, "spool_size": 0}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            result = await client.is_daemon_running()

            assert result is True

    async def test_is_daemon_running_returns_false_when_socket_not_found(self) -> None:
        """Test is_daemon_running() returns False when socket doesn't exist."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=FileNotFoundError("Socket not found"),
        ):
            client = EmitClient(socket_path="/tmp/nonexistent.sock")
            result = await client.is_daemon_running()

            assert result is False

    async def test_is_daemon_running_returns_false_when_connection_refused(
        self,
    ) -> None:
        """Test is_daemon_running() returns False when connection refused."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            result = await client.is_daemon_running()

            assert result is False

    async def test_is_daemon_running_returns_false_on_timeout(self) -> None:
        """Test is_daemon_running() returns False on timeout."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=TimeoutError("Connection timeout"),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            result = await client.is_daemon_running()

            assert result is False

    async def test_is_daemon_running_returns_false_on_unexpected_error(self) -> None:
        """Test is_daemon_running() returns False on any unexpected error."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=RuntimeError("Unexpected error"),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            result = await client.is_daemon_running()

            assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmitClientSocketErrors:
    """Test socket error handling scenarios."""

    async def test_socket_not_found_raises_emit_client_error(self) -> None:
        """Test that emit() raises EmitClientError when socket doesn't exist."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=FileNotFoundError("Socket not found"),
        ):
            client = EmitClient(socket_path="/tmp/nonexistent.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert "not found" in str(exc_info.value).lower()
            assert exc_info.value.reason == "socket_not_found"

    async def test_connection_refused_raises_emit_client_error(self) -> None:
        """Test that emit() raises EmitClientError when connection refused."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert "refused" in str(exc_info.value).lower()
            assert exc_info.value.reason == "connection_refused"

    async def test_permission_denied_raises_emit_client_error(self) -> None:
        """Test that emit() raises EmitClientError when permission denied."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=PermissionError("Permission denied"),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert "permission" in str(exc_info.value).lower()
            assert exc_info.value.reason == "permission_denied"

    async def test_connection_timeout_raises_emit_client_error(self) -> None:
        """Test that emit() raises EmitClientError on connection timeout."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=TimeoutError("Connection timeout"),
        ):
            client = EmitClient(socket_path="/tmp/test.sock", timeout=1.0)

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert "timeout" in str(exc_info.value).lower()
            assert exc_info.value.reason == "connection_timeout"

    async def test_response_timeout_raises_emit_client_error(self) -> None:
        """Test that emit() raises EmitClientError on response timeout."""
        mock_reader = AsyncMock()
        # readline raises TimeoutError to simulate response timeout
        mock_reader.readline = AsyncMock(side_effect=TimeoutError("Read timeout"))

        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock", timeout=1.0)

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert "timeout" in str(exc_info.value).lower()
            assert exc_info.value.reason == "response_timeout"

    async def test_connection_reset_raises_emit_client_error(self) -> None:
        """Test that emit() raises EmitClientError when connection reset."""
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(side_effect=ConnectionResetError())

        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert exc_info.value.reason == "connection_reset"

    async def test_broken_pipe_raises_emit_client_error(self) -> None:
        """Test that emit() raises EmitClientError on broken pipe."""
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(side_effect=BrokenPipeError())

        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert exc_info.value.reason == "broken_pipe"

    async def test_connection_closed_by_daemon_raises_emit_client_error(self) -> None:
        """Test that empty response (connection closed) raises EmitClientError."""
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=b"")  # Empty = closed

        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert exc_info.value.reason == "connection_closed"

    async def test_invalid_json_response_raises_emit_client_error(self) -> None:
        """Test that invalid JSON response raises EmitClientError."""
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=b"not valid json\n")

        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert exc_info.value.reason == "invalid_json"

    async def test_non_dict_json_response_raises_emit_client_error(self) -> None:
        """Test that non-dict JSON response raises EmitClientError."""
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=b'["array", "not", "dict"]\n')

        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert exc_info.value.reason == "invalid_response"

    async def test_os_error_raises_emit_client_error(self) -> None:
        """Test that generic OSError raises EmitClientError."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=OSError("Generic OS error"),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")

            with pytest.raises(EmitClientError) as exc_info:
                await client.emit("test.event", {"key": "value"})

            assert exc_info.value.reason == "os_error"

    async def test_timeout_respects_configured_value(self) -> None:
        """Test that timeout configuration is respected."""
        client = EmitClient(socket_path="/tmp/test.sock", timeout=2.5)

        assert client.timeout == 2.5

    async def test_default_timeout_value(self) -> None:
        """Test that default timeout is 5.0 seconds."""
        client = EmitClient(socket_path="/tmp/test.sock")

        assert client.timeout == EmitClient.DEFAULT_TIMEOUT
        assert client.timeout == 5.0


@pytest.mark.unit
class TestEmitClientSyncWrappers:
    """Test synchronous wrapper methods."""

    def test_emit_sync_works_without_event_loop(self) -> None:
        """Test emit_sync() works without running event loop."""
        response = {"status": "queued", "event_id": "sync-id-123"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            event_id = client.emit_sync("test.event", {"key": "value"})

            assert event_id == "sync-id-123"

    def test_ping_sync_works_without_event_loop(self) -> None:
        """Test ping_sync() works without running event loop."""
        response = {"status": "ok", "queue_size": 3, "spool_size": 7}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            result = client.ping_sync()

            assert result["status"] == "ok"
            assert result["queue_size"] == 3
            assert result["spool_size"] == 7

    def test_is_daemon_running_sync_works_without_event_loop(self) -> None:
        """Test is_daemon_running_sync() works without running event loop."""
        response = {"status": "ok", "queue_size": 0, "spool_size": 0}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            result = client.is_daemon_running_sync()

            assert result is True

    def test_emit_sync_raises_emit_client_error_on_failure(self) -> None:
        """Test emit_sync() raises EmitClientError on failure."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=FileNotFoundError("Socket not found"),
        ):
            client = EmitClient(socket_path="/tmp/nonexistent.sock")

            with pytest.raises(EmitClientError) as exc_info:
                client.emit_sync("test.event", {"key": "value"})

            assert exc_info.value.reason == "socket_not_found"

    @pytest.mark.asyncio
    async def test_sync_methods_raise_in_async_context(self) -> None:
        """Test sync methods raise RuntimeError when called from async context."""
        client = EmitClient(socket_path="/tmp/test.sock")

        with pytest.raises(RuntimeError) as exc_info:
            client.emit_sync("test.event", {"key": "value"})

        assert "async context" in str(exc_info.value).lower()

    def test_emit_sync_disconnects_after_operation(self) -> None:
        """Test emit_sync() properly disconnects after operation."""
        response = {"status": "queued", "event_id": "test-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            client.emit_sync("test.event", {"key": "value"})

            # Connection should be closed after sync operation
            mock_writer.close.assert_called()


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmitEventHelper:
    """Test emit_event() convenience function."""

    async def test_emit_event_creates_client_and_emits(self) -> None:
        """Test emit_event() convenience function creates client and emits."""
        response = {"status": "queued", "event_id": "helper-event-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            event_id = await emit_event(
                "prompt.submitted",
                {"prompt_id": "test-123"},
            )

            assert event_id == "helper-event-id"

    async def test_emit_event_uses_custom_socket_path(self) -> None:
        """Test emit_event() uses custom socket path."""
        response = {"status": "queued", "event_id": "custom-path-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ) as mock_connect:
            await emit_event(
                "test.event",
                {"key": "value"},
                socket_path=Path("/custom/path.sock"),
            )

            # Verify the custom path was used
            mock_connect.assert_called_once()
            call_args = mock_connect.call_args[0]
            assert "/custom/path.sock" in str(call_args[0])

    async def test_emit_event_uses_custom_timeout(self) -> None:
        """Test emit_event() passes custom timeout to client."""
        response = {"status": "queued", "event_id": "timeout-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            # Custom timeout should not affect success path
            event_id = await emit_event(
                "test.event",
                {"key": "value"},
                timeout=10.0,
            )

            assert event_id == "timeout-id"

    async def test_emit_event_raises_on_error(self) -> None:
        """Test emit_event() raises EmitClientError on failure."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=FileNotFoundError("Socket not found"),
        ):
            with pytest.raises(EmitClientError):
                await emit_event("test.event", {"key": "value"})


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmitEventWithFallback:
    """Test emit_event_with_fallback() with fallback logic."""

    async def test_uses_daemon_when_available(self) -> None:
        """Test emit_event_with_fallback() uses daemon when available."""
        response = {"status": "queued", "event_id": "daemon-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)
        fallback_called = False

        async def mock_fallback(event_type: str, payload: dict[str, object]) -> str:
            nonlocal fallback_called
            fallback_called = True
            return "fallback-id"

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            event_id = await emit_event_with_fallback(
                "test.event",
                {"key": "value"},
                fallback=mock_fallback,
            )

            assert event_id == "daemon-id"
            assert fallback_called is False

    async def test_calls_fallback_when_daemon_unavailable(self) -> None:
        """Test emit_event_with_fallback() calls fallback when daemon unavailable."""
        fallback_event_type = None
        fallback_payload = None

        async def mock_fallback(event_type: str, payload: dict[str, object]) -> str:
            nonlocal fallback_event_type, fallback_payload
            fallback_event_type = event_type
            fallback_payload = payload
            return "fallback-event-id"

        with patch(
            "asyncio.open_unix_connection",
            side_effect=FileNotFoundError("Socket not found"),
        ):
            event_id = await emit_event_with_fallback(
                "test.event",
                {"key": "value"},
                fallback=mock_fallback,
            )

            assert event_id == "fallback-event-id"
            assert fallback_event_type == "test.event"
            assert fallback_payload == {"key": "value"}

    async def test_fallback_called_on_connection_refused(self) -> None:
        """Test fallback is called when connection is refused."""

        async def mock_fallback(event_type: str, payload: dict[str, object]) -> str:
            return "fallback-refused-id"

        with patch(
            "asyncio.open_unix_connection",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            event_id = await emit_event_with_fallback(
                "test.event",
                {"key": "value"},
                fallback=mock_fallback,
            )

            assert event_id == "fallback-refused-id"

    async def test_fallback_called_on_connection_timeout(self) -> None:
        """Test fallback is called on connection timeout."""

        async def mock_fallback(event_type: str, payload: dict[str, object]) -> str:
            return "fallback-timeout-id"

        with patch(
            "asyncio.open_unix_connection",
            side_effect=TimeoutError("Timeout"),
        ):
            event_id = await emit_event_with_fallback(
                "test.event",
                {"key": "value"},
                fallback=mock_fallback,
            )

            assert event_id == "fallback-timeout-id"

    async def test_fallback_called_on_permission_denied(self) -> None:
        """Test fallback is called when permission denied."""

        async def mock_fallback(event_type: str, payload: dict[str, object]) -> str:
            return "fallback-perm-id"

        with patch(
            "asyncio.open_unix_connection",
            side_effect=PermissionError("Permission denied"),
        ):
            event_id = await emit_event_with_fallback(
                "test.event",
                {"key": "value"},
                fallback=mock_fallback,
            )

            assert event_id == "fallback-perm-id"

    async def test_raises_without_fallback_when_daemon_unavailable(self) -> None:
        """Test raises EmitClientError when daemon unavailable and no fallback."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=FileNotFoundError("Socket not found"),
        ):
            with pytest.raises(EmitClientError):
                await emit_event_with_fallback(
                    "test.event",
                    {"key": "value"},
                    fallback=None,
                )

    async def test_returns_result_from_fallback(self) -> None:
        """Test that fallback return value is properly returned."""
        expected_fallback_id = "custom-fallback-uuid-12345"

        async def mock_fallback(event_type: str, payload: dict[str, object]) -> str:
            return expected_fallback_id

        with patch(
            "asyncio.open_unix_connection",
            side_effect=FileNotFoundError("Socket not found"),
        ):
            event_id = await emit_event_with_fallback(
                "test.event",
                {"key": "value"},
                fallback=mock_fallback,
            )

            assert event_id == expected_fallback_id

    async def test_re_raises_non_connection_errors(self) -> None:
        """Test that non-connection errors are re-raised even with fallback."""
        response = {"status": "error", "reason": "payload_validation_failed"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        async def mock_fallback(event_type: str, payload: dict[str, object]) -> str:
            return "should-not-be-called"

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            # Daemon is available but rejects the event - should not use fallback
            with pytest.raises(EmitClientError) as exc_info:
                await emit_event_with_fallback(
                    "test.event",
                    {"invalid": "payload"},
                    fallback=mock_fallback,
                )

            assert exc_info.value.reason == "payload_validation_failed"


@pytest.mark.unit
class TestEmitClientError:
    """Test EmitClientError exception class structure."""

    def test_error_has_message_attribute(self) -> None:
        """Test EmitClientError has accessible message."""
        error = EmitClientError("Test error message")
        assert str(error) == "Test error message"

    def test_error_has_reason_attribute(self) -> None:
        """Test EmitClientError has reason attribute."""
        error = EmitClientError("Test error", reason="test_reason")
        assert error.reason == "test_reason"

    def test_error_reason_defaults_to_none(self) -> None:
        """Test EmitClientError reason defaults to None."""
        error = EmitClientError("Test error")
        assert error.reason is None

    def test_error_string_representation(self) -> None:
        """Test EmitClientError string representation includes message."""
        error = EmitClientError("Connection failed", reason="socket_not_found")
        assert "Connection failed" in str(error)

    def test_error_inherits_from_exception(self) -> None:
        """Test EmitClientError inherits from Exception."""
        error = EmitClientError("Test")
        assert isinstance(error, Exception)

    def test_error_can_be_caught_as_exception(self) -> None:
        """Test EmitClientError can be caught as generic Exception."""
        try:
            raise EmitClientError("Test error", reason="test")
        except Exception as e:
            assert isinstance(e, EmitClientError)
            assert e.reason == "test"


@pytest.mark.unit
class TestEmitClientProperties:
    """Test EmitClient property accessors."""

    def test_socket_path_property(self) -> None:
        """Test socket_path property returns configured path."""
        path = Path("/custom/socket.sock")
        client = EmitClient(socket_path=path)
        assert client.socket_path == path

    def test_socket_path_accepts_string(self) -> None:
        """Test socket_path accepts string and converts to Path."""
        client = EmitClient(socket_path="/string/path.sock")
        assert client.socket_path == Path("/string/path.sock")

    def test_timeout_property(self) -> None:
        """Test timeout property returns configured value."""
        client = EmitClient(timeout=15.0)
        assert client.timeout == 15.0

    def test_default_socket_path(self) -> None:
        """Test default socket path is set correctly."""
        client = EmitClient()
        assert client.socket_path == EmitClient.DEFAULT_SOCKET_PATH
        assert client.socket_path == Path("/tmp/omniclaude-emit.sock")


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmitClientConnectionManagement:
    """Test connection lifecycle management."""

    async def test_connection_reused_in_context_manager(self) -> None:
        """Test that connection is reused for multiple emits in context manager."""
        responses = [
            {"status": "queued", "event_id": "id-1"},
            {"status": "queued", "event_id": "id-2"},
        ]
        call_count = 0

        async def mock_readline() -> bytes:
            nonlocal call_count
            response = responses[call_count]
            call_count += 1
            return (json.dumps(response) + "\n").encode("utf-8")

        mock_reader = AsyncMock()
        mock_reader.readline = mock_readline

        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ) as mock_connect:
            async with EmitClient(socket_path="/tmp/test.sock") as client:
                id1 = await client.emit("event.one", {"n": 1})
                id2 = await client.emit("event.two", {"n": 2})

                assert id1 == "id-1"
                assert id2 == "id-2"

            # Connection should only be opened once
            mock_connect.assert_called_once()

    async def test_disconnect_closes_writer(self) -> None:
        """Test that disconnect properly closes the writer."""
        response = {"status": "queued", "event_id": "test-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            async with EmitClient(socket_path="/tmp/test.sock") as client:
                await client.emit("test.event", {"key": "value"})

            mock_writer.close.assert_called_once()
            mock_writer.wait_closed.assert_awaited_once()

    async def test_disconnect_safe_to_call_multiple_times(self) -> None:
        """Test that disconnect can be called multiple times safely."""
        response = {"status": "queued", "event_id": "test-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ):
            client = EmitClient(socket_path="/tmp/test.sock")
            await client.emit("test.event", {"key": "value"})

            # Call disconnect multiple times - should not raise
            await client._disconnect()
            await client._disconnect()
            await client._disconnect()

            # close should only be called once (first disconnect)
            assert mock_writer.close.call_count == 1

    async def test_lazy_connection_on_first_emit(self) -> None:
        """Test that connection is established lazily on first emit."""
        response = {"status": "queued", "event_id": "lazy-id"}
        mock_reader, mock_writer = create_mock_reader_writer(response)

        with patch(
            "asyncio.open_unix_connection",
            return_value=(mock_reader, mock_writer),
        ) as mock_connect:
            client = EmitClient(socket_path="/tmp/test.sock")

            # Connection not established yet
            mock_connect.assert_not_called()

            # First emit establishes connection
            await client.emit("test.event", {"key": "value"})
            mock_connect.assert_called_once()

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HttpRestHandler.

Comprehensive test suite covering initialization, GET/POST operations,
error handling, describe, and lifecycle management.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from omnibase_core.enums.enum_handler_type import EnumHandlerType

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_http import HttpRestHandler
from tests.helpers import (
    DeterministicClock,
    DeterministicIdGenerator,
    filter_handler_warnings,
)


def create_mock_streaming_response(
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    body_bytes: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock httpx.Response that supports streaming iteration.

    Args:
        status_code: HTTP status code
        headers: Response headers (content-type will be added if not present)
        body_bytes: The response body as bytes
        content_type: Content-Type header value (used if headers doesn't have it)

    Returns:
        MagicMock configured to behave like httpx.Response with streaming support
    """
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code

    # Build headers dict with content-type
    response_headers = headers or {}
    if "content-type" not in response_headers:
        response_headers["content-type"] = content_type
    mock_response.headers = response_headers

    # Create an async iterator for aiter_bytes
    async def aiter_bytes_impl(chunk_size: int = 8192) -> AsyncIterator[bytes]:
        """Yield body_bytes in chunks."""
        for i in range(0, len(body_bytes), chunk_size):
            yield body_bytes[i : i + chunk_size]
        # Handle empty body case - still need to yield nothing
        if len(body_bytes) == 0:
            return

    mock_response.aiter_bytes = aiter_bytes_impl

    return mock_response


@asynccontextmanager
async def mock_stream_context(
    mock_response: MagicMock,
) -> AsyncIterator[MagicMock]:
    """Create an async context manager that yields the mock response.

    This simulates httpx.AsyncClient.stream() behavior.
    """
    yield mock_response


class TestHttpRestHandlerInitialization:
    """Test suite for HttpRestHandler initialization."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    def test_handler_init_default_state(self, handler: HttpRestHandler) -> None:
        """Test handler initializes in uninitialized state."""
        assert handler._initialized is False
        assert handler._client is None
        assert handler._timeout == 30.0

    def test_handler_type_returns_http(self, handler: HttpRestHandler) -> None:
        """Test handler_type property returns EnumHandlerType.HTTP."""
        assert handler.handler_type == EnumHandlerType.HTTP

    @pytest.mark.asyncio
    async def test_initialize_with_empty_config(self, handler: HttpRestHandler) -> None:
        """Test handler initializes with empty config (uses defaults)."""
        await handler.initialize({})

        assert handler._initialized is True
        assert handler._client is not None
        assert handler._timeout == 30.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_config_dict(self, handler: HttpRestHandler) -> None:
        """Test handler initializes with config dict (config ignored in MVP)."""
        config: dict[str, object] = {"timeout": 60.0, "custom_option": "value"}
        await handler.initialize(config)

        # MVP ignores config, uses fixed 30s timeout
        assert handler._initialized is True
        assert handler._timeout == 30.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_creates_async_client(
        self, handler: HttpRestHandler
    ) -> None:
        """Test initialize creates httpx.AsyncClient with correct timeout."""
        await handler.initialize({})

        assert isinstance(handler._client, httpx.AsyncClient)
        # Verify timeout is set (30s default)
        assert handler._client.timeout.connect == 30.0

        await handler.shutdown()


class TestHttpRestHandlerGetOperations:
    """Test suite for HTTP GET operations."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_get_successful_response(self, handler: HttpRestHandler) -> None:
        """Test successful GET request returns correct response structure."""
        await handler.initialize({})

        # Create mock response with streaming support
        import json as json_module

        body_data = {"data": "test_value"}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json", "x-custom": "value"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            correlation_id = uuid4()
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/resource"},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["status"] == "success"
            payload = result["payload"]
            assert payload["status_code"] == 200
            assert payload["body"] == {"data": "test_value"}
            assert result["correlation_id"] == str(correlation_id)

            mock_stream.assert_called_once_with(
                "GET",
                "https://api.example.com/resource",
                headers={},
                content=None,
                json=None,
            )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_with_custom_headers(self, handler: HttpRestHandler) -> None:
        """Test GET request passes custom headers correctly."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "text/plain"},
            body_bytes=b"OK",
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {
                    "url": "https://api.example.com/status",
                    "headers": {
                        "Authorization": "Bearer token123",
                        "X-Request-ID": "req-456",
                    },
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            mock_stream.assert_called_once_with(
                "GET",
                "https://api.example.com/status",
                headers={"Authorization": "Bearer token123", "X-Request-ID": "req-456"},
                content=None,
                json=None,
            )

            assert result["payload"]["body"] == "OK"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_with_query_params_in_url(self, handler: HttpRestHandler) -> None:
        """Test GET request with query parameters in URL."""
        await handler.initialize({})

        import json as json_module

        body_data = {"items": [1, 2, 3]}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {
                    "url": "https://api.example.com/items?page=1&limit=10&filter=active",
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            mock_stream.assert_called_once_with(
                "GET",
                "https://api.example.com/items?page=1&limit=10&filter=active",
                headers={},
                content=None,
                json=None,
            )

            assert result["payload"]["body"] == {"items": [1, 2, 3]}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_text_response(self, handler: HttpRestHandler) -> None:
        """Test GET request with text/plain response."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "text/plain; charset=utf-8"},
            body_bytes=b"Hello, World!",
            content_type="text/plain; charset=utf-8",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/hello"},
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["payload"]["body"] == "Hello, World!"
            assert result["payload"]["status_code"] == 200

        await handler.shutdown()


class TestHttpRestHandlerPostOperations:
    """Test suite for HTTP POST operations."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_post_with_json_body(self, handler: HttpRestHandler) -> None:
        """Test POST request with JSON body.

        Note: Dict bodies are pre-serialized during size validation to avoid
        double serialization. The serialized bytes are passed via content=
        instead of json=, with Content-Type header set explicitly.
        """
        await handler.initialize({})

        import json as json_module

        body_data = {"id": 123, "created": True}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=201,
            headers={"content-type": "application/json"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            request_body = {"name": "John", "email": "john@example.com"}
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/users",
                    "body": request_body,
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            # Dict bodies are pre-serialized to avoid double serialization.
            # Expect content= with serialized bytes and Content-Type header.
            expected_content = json_module.dumps(request_body).encode("utf-8")
            mock_stream.assert_called_once_with(
                "POST",
                "https://api.example.com/users",
                headers={"Content-Type": "application/json"},
                content=expected_content,
                json=None,
            )

            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 201
            assert result["payload"]["body"] == {"id": 123, "created": True}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_string_body(self, handler: HttpRestHandler) -> None:
        """Test POST request with string body."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "text/plain"},
            body_bytes=b"Received",
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/message",
                    "body": "Hello from client",
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            mock_stream.assert_called_once_with(
                "POST",
                "https://api.example.com/message",
                headers={},
                content="Hello from client",
                json=None,
            )

            assert result["payload"]["body"] == "Received"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_no_body(self, handler: HttpRestHandler) -> None:
        """Test POST request with no body."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=204,
            headers={"content-type": "text/plain"},
            body_bytes=b"",
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {"url": "https://api.example.com/trigger"},
            }

            output = await handler.execute(envelope)
            result = output.result

            mock_stream.assert_called_once_with(
                "POST",
                "https://api.example.com/trigger",
                headers={},
                content=None,
                json=None,
            )

            assert result["payload"]["status_code"] == 204

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_custom_headers(self, handler: HttpRestHandler) -> None:
        """Test POST request with custom headers.

        Note: Dict bodies are pre-serialized during size validation. When
        Content-Type is already set in custom headers, it's preserved.
        """
        await handler.initialize({})

        import json as json_module

        body_data = {"success": True}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            request_body = {"value": 42}
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/data",
                    "headers": {
                        "Content-Type": "application/json",
                        "X-API-Key": "secret-key-123",
                    },
                    "body": request_body,
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            # Dict bodies are pre-serialized. Content-Type is preserved from
            # custom headers since it's already set (case-insensitive check).
            expected_content = json_module.dumps(request_body).encode("utf-8")
            mock_stream.assert_called_once_with(
                "POST",
                "https://api.example.com/data",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": "secret-key-123",
                },
                content=expected_content,
                json=None,
            )

            assert result["payload"]["body"] == {"success": True}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_list_body_serialized_to_json(
        self, handler: HttpRestHandler
    ) -> None:
        """Test POST with list body gets JSON serialized."""
        await handler.initialize({})

        import json as json_module

        body_data = {"processed": 3}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/batch",
                    "body": [1, 2, 3],  # List body, not dict or string
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            # List body uses content= with json.dumps()
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            assert call_args.kwargs["content"] == "[1, 2, 3]"

            assert result["payload"]["body"] == {"processed": 3}

        await handler.shutdown()


class TestHttpRestHandlerErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_timeout_error_raises_infra_timeout(
        self, handler: HttpRestHandler
    ) -> None:
        """Test timeout error is converted to InfraTimeoutError."""
        await handler.initialize({})

        @asynccontextmanager
        async def raise_timeout() -> AsyncIterator[MagicMock]:
            raise httpx.TimeoutException("Connection timed out")
            yield MagicMock()  # Never reached, but makes type checker happy

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = raise_timeout()

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://slow.example.com/api"},
            }

            with pytest.raises(InfraTimeoutError) as exc_info:
                await handler.execute(envelope)

            assert "timed out" in str(exc_info.value)
            assert "30" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_connection_error_raises_infra_connection(
        self, handler: HttpRestHandler
    ) -> None:
        """Test connection error is converted to InfraConnectionError."""
        await handler.initialize({})

        @asynccontextmanager
        async def raise_connect_error() -> AsyncIterator[MagicMock]:
            raise httpx.ConnectError("Connection refused")
            yield MagicMock()  # Never reached, but makes type checker happy

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = raise_connect_error()

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://unreachable.example.com/api"},
            }

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(envelope)

            assert "Failed to connect" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unsupported_operation_put_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test http.put operation raises ProtocolConfigurationError (not supported)."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.put",
            "payload": {"url": "https://api.example.com/resource/123"},
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "http.put" in str(exc_info.value)
        assert "not supported" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unsupported_operation_delete_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test http.delete operation raises ProtocolConfigurationError (not supported)."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.delete",
            "payload": {"url": "https://api.example.com/resource/123"},
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "http.delete" in str(exc_info.value)
        assert "not supported" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unsupported_operation_patch_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test http.patch operation raises ProtocolConfigurationError (not supported)."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.patch",
            "payload": {"url": "https://api.example.com/resource/123"},
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "http.patch" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_url_field_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test missing URL field raises ProtocolConfigurationError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"headers": {"X-Test": "value"}},  # No URL
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "url" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_empty_url_field_raises_error(self, handler: HttpRestHandler) -> None:
        """Test empty URL field raises ProtocolConfigurationError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": ""},
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "url" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_operation_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test missing operation field raises ProtocolConfigurationError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "payload": {"url": "https://example.com"},
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "operation" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_payload_raises_error(self, handler: HttpRestHandler) -> None:
        """Test missing payload field raises ProtocolConfigurationError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "payload" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_headers_type_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test invalid headers type raises ProtocolConfigurationError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {
                "url": "https://example.com",
                "headers": "not-a-dict",  # Invalid type
            },
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        assert "headers" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_http_status_error_returns_response(
        self, handler: HttpRestHandler
    ) -> None:
        """Test non-2xx HTTP responses still return successfully (not an exception).

        Note: With streaming, we don't get HTTPStatusError - instead we get
        the response directly and check the status code. HTTP 4xx/5xx are not
        exceptions, they're valid HTTP responses.
        """
        await handler.initialize({})

        import json as json_module

        body_data = {"error": "Not found"}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=404,
            headers={"content-type": "application/json"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/missing"},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Should return the error response, not raise
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 404
            assert result["payload"]["body"] == {"error": "Not found"}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_generic_http_error_raises_connection_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test generic HTTPError raises InfraConnectionError."""
        await handler.initialize({})

        @asynccontextmanager
        async def raise_http_error() -> AsyncIterator[MagicMock]:
            raise httpx.HTTPError("Unknown HTTP error")
            yield MagicMock()  # Never reached

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = raise_http_error()

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/broken"},
            }

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(envelope)

            assert "HTTP error" in str(exc_info.value)

        await handler.shutdown()


class TestHttpRestHandlerDescribe:
    """Test suite for describe operations."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    def test_describe_returns_handler_metadata(self, handler: HttpRestHandler) -> None:
        """Test describe returns correct handler metadata."""
        description = handler.describe()

        assert description["handler_type"] == "http"
        assert description["timeout_seconds"] == 30.0
        assert description["version"] == "0.1.0-mvp"
        assert description["initialized"] is False

    def test_describe_lists_supported_operations(
        self, handler: HttpRestHandler
    ) -> None:
        """Test describe lists supported operations."""
        description = handler.describe()

        assert "supported_operations" in description
        operations = description["supported_operations"]

        assert "http.get" in operations
        assert "http.post" in operations
        assert len(operations) == 2

    @pytest.mark.asyncio
    async def test_describe_reflects_initialized_state(
        self, handler: HttpRestHandler
    ) -> None:
        """Test describe shows correct initialized state."""
        assert handler.describe()["initialized"] is False

        await handler.initialize({})
        assert handler.describe()["initialized"] is True

        await handler.shutdown()
        assert handler.describe()["initialized"] is False


class TestHttpRestHandlerLifecycle:
    """Test suite for lifecycle management."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self, handler: HttpRestHandler) -> None:
        """Test shutdown closes the HTTP client properly."""
        await handler.initialize({})

        client = handler._client
        assert client is not None

        with patch.object(client, "aclose", new_callable=AsyncMock) as mock_close:
            await handler.shutdown()

            mock_close.assert_called_once()
            assert handler._client is None
            assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_execute_after_shutdown_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test execute after shutdown raises RuntimeHostError."""
        await handler.initialize({})
        await handler.shutdown()

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_before_initialize_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test execute before initialize raises RuntimeHostError."""
        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_multiple_shutdown_calls_safe(self, handler: HttpRestHandler) -> None:
        """Test multiple shutdown calls are safe (idempotent)."""
        await handler.initialize({})
        await handler.shutdown()
        await handler.shutdown()  # Second call should not raise

        assert handler._initialized is False
        assert handler._client is None

    @pytest.mark.asyncio
    async def test_reinitialize_after_shutdown(self, handler: HttpRestHandler) -> None:
        """Test handler can be reinitialized after shutdown."""
        await handler.initialize({})
        await handler.shutdown()

        assert handler._initialized is False

        await handler.initialize({})

        assert handler._initialized is True
        assert handler._client is not None

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_called_once_per_lifecycle(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that initialize creates client exactly once.

        Acceptance criteria for OMN-252: Asserts handler initialized exactly once.
        Each call to initialize() should create a fresh client instance.
        """
        # First initialize
        await handler.initialize({})
        first_client = handler._client
        assert first_client is not None

        # Second initialize should create new client (reinitialize behavior)
        await handler.initialize({})
        second_client = handler._client
        assert second_client is not None

        # Verify we got a new client (not reusing old one)
        # This confirms initialize() creates resources fresh each time
        assert first_client is not second_client, (
            "initialize() should create new client instance"
        )
        await handler.shutdown()


class TestHttpRestHandlerCorrelationId:
    """Test suite for correlation ID handling."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_uuid(
        self, handler: HttpRestHandler
    ) -> None:
        """Test correlation ID extracted from envelope as UUID.

        Uses DeterministicIdGenerator for predictable, reproducible test behavior.
        """
        await handler.initialize({})

        # Use deterministic ID generator for predictable testing
        id_gen = DeterministicIdGenerator(seed=100)
        correlation_id = id_gen.next_uuid()
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b"{}",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result

            # Verify deterministic UUID is properly returned (as string in result)
            assert result["correlation_id"] == str(correlation_id)
            # With seed=100, first UUID has int value 101
            assert correlation_id.int == 101

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_string(
        self, handler: HttpRestHandler
    ) -> None:
        """Test correlation ID extracted from envelope as string."""
        await handler.initialize({})

        correlation_id = str(uuid4())
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b"{}",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result

            # String correlation_id is converted to UUID by handler,
            # then back to string in the result
            assert result["correlation_id"] == correlation_id

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_generated_when_missing(
        self, handler: HttpRestHandler
    ) -> None:
        """Test correlation ID generated when not in envelope."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b"{}",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Should have a generated UUID (returned as string in result)
            assert "correlation_id" in result
            # Verify it's a valid UUID string
            assert isinstance(result["correlation_id"], str)
            UUID(result["correlation_id"])  # Should not raise

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_invalid_string_generates_new(
        self, handler: HttpRestHandler
    ) -> None:
        """Test invalid correlation ID string generates new UUID."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b"{}",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
                "correlation_id": "not-a-valid-uuid",
            }

            output = await handler.execute(envelope)
            result = output.result

            # Should have a generated UUID (not the invalid string, returned as string)
            assert "correlation_id" in result
            generated_id = result["correlation_id"]
            assert isinstance(generated_id, str)
            assert generated_id != "not-a-valid-uuid"
            UUID(generated_id)  # Should parse as valid UUID

        await handler.shutdown()


class TestHttpRestHandlerResponseParsing:
    """Test suite for response parsing."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_json_response_parsed(self, handler: HttpRestHandler) -> None:
        """Test JSON response is parsed correctly."""
        await handler.initialize({})

        import json as json_module

        body_data = {"key": "value", "nested": {"a": 1}}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json; charset=utf-8"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["payload"]["body"] == {"key": "value", "nested": {"a": 1}}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_text(self, handler: HttpRestHandler) -> None:
        """Test invalid JSON response falls back to text."""
        await handler.initialize({})

        # Create a response with content-type json but invalid JSON body
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b"Not valid JSON {",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["payload"]["body"] == "Not valid JSON {"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_non_json_content_type_returns_text(
        self, handler: HttpRestHandler
    ) -> None:
        """Test non-JSON content type returns text body."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            body_bytes=b"<html><body>Hello</body></html>",
            content_type="text/html; charset=utf-8",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["payload"]["body"] == "<html><body>Hello</body></html>"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_response_headers_included(self, handler: HttpRestHandler) -> None:
        """Test response headers are included in result."""
        await handler.initialize({})

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "x-request-id": "req-123",
                "x-rate-limit-remaining": "99",
            },
            body_bytes=b"{}",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["payload"]["headers"]["content-type"] == "application/json"
            assert result["payload"]["headers"]["x-request-id"] == "req-123"
            assert result["payload"]["headers"]["x-rate-limit-remaining"] == "99"

        await handler.shutdown()


class TestHttpRestHandlerSizeLimits:
    """Test suite for request/response size limits."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    def test_default_size_limits(self, handler: HttpRestHandler) -> None:
        """Test default size limits are set correctly."""
        assert handler._max_request_size == 10 * 1024 * 1024  # 10 MB
        assert handler._max_response_size == 50 * 1024 * 1024  # 50 MB

    @pytest.mark.asyncio
    async def test_configurable_size_limits(self, handler: HttpRestHandler) -> None:
        """Test size limits can be configured via initialize()."""
        config: dict[str, object] = {
            "max_request_size": 1024,  # 1 KB
            "max_response_size": 2048,  # 2 KB
        }
        await handler.initialize(config)

        assert handler._max_request_size == 1024
        assert handler._max_response_size == 2048

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_size_validation_string_body(
        self, handler: HttpRestHandler
    ) -> None:
        """Test request size validation with string body."""
        config: dict[str, object] = {"max_request_size": 10}  # 10 bytes
        await handler.initialize(config)

        # String body that exceeds limit
        envelope: dict[str, object] = {
            "operation": "http.post",
            "payload": {
                "url": "https://example.com",
                "body": "This string is definitely longer than 10 bytes",
            },
        }

        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(envelope)

        # Error message uses sanitized size categories instead of exact byte values
        assert "exceeds configured limit" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_size_validation_dict_body(
        self, handler: HttpRestHandler
    ) -> None:
        """Test request size validation with dict body (JSON serialized)."""
        config: dict[str, object] = {"max_request_size": 10}  # 10 bytes
        await handler.initialize(config)

        # Dict body that exceeds limit when serialized
        envelope: dict[str, object] = {
            "operation": "http.post",
            "payload": {
                "url": "https://example.com",
                "body": {"key": "value", "another": "field"},
            },
        }

        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(envelope)

        # Error message uses sanitized size categories instead of exact byte values
        assert "exceeds configured limit" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_size_validation_bytes_body(
        self, handler: HttpRestHandler
    ) -> None:
        """Test request size validation with bytes body."""
        config: dict[str, object] = {"max_request_size": 5}  # 5 bytes
        await handler.initialize(config)

        # Testing the internal _validate_request_size() method directly with bytes body
        # is acceptable because:
        # 1. Bytes bodies ARE valid inputs (can be passed via payload['body'] in execute())
        # 2. Direct method testing validates the size calculation logic efficiently
        # 3. Avoids complex HTTP mocking that would be needed for end-to-end testing
        # 4. The method is part of the public validation contract for the adapter
        correlation_id = uuid4()

        # Test the internal validation method with bytes
        with pytest.raises(InfraUnavailableError) as exc_info:
            handler._validate_request_size(b"123456", correlation_id)

        # Error message uses sanitized size categories instead of exact byte values
        assert "exceeds configured limit" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_size_exceeds_limit_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that request exceeding size limit raises InfraUnavailableError."""
        config: dict[str, object] = {"max_request_size": 100}
        await handler.initialize(config)

        large_body = "x" * 200  # 200 bytes, exceeds 100 byte limit
        envelope: dict[str, object] = {
            "operation": "http.post",
            "payload": {
                "url": "https://example.com",
                "body": large_body,
            },
        }

        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(envelope)

        error_msg = str(exc_info.value)
        # Error message uses sanitized size categories instead of exact byte values
        assert "exceeds configured limit" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_response_size_exceeds_limit_raises_error(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that response exceeding size limit raises InfraUnavailableError."""
        config: dict[str, object] = {"max_response_size": 50}  # 50 bytes
        await handler.initialize(config)

        # Create a streaming response with body > 50 bytes
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "text/plain"},
            body_bytes=b"x" * 100,  # 100 bytes, exceeds 50 byte limit
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            error_msg = str(exc_info.value)
            # Error message uses sanitized size categories instead of exact byte values
            assert "exceeds configured limit" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_none_body_skips_validation(self, handler: HttpRestHandler) -> None:
        """Test that None body skips size validation."""
        config: dict[str, object] = {"max_request_size": 1}  # 1 byte - very small
        await handler.initialize(config)

        mock_response = create_mock_streaming_response(
            status_code=204,
            headers={"content-type": "text/plain"},
            body_bytes=b"",
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            # POST with no body should not raise size limit error
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {"url": "https://example.com"},  # No body
            }

            output = await handler.execute(envelope)
            result = output.result
            assert result["status"] == "success"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_within_limit_succeeds(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that request within size limit succeeds."""
        config: dict[str, object] = {"max_request_size": 1000}  # 1000 bytes
        await handler.initialize(config)

        import json as json_module

        body_data = {"success": True}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=body_bytes,
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            small_body = "x" * 50  # 50 bytes, under 1000 byte limit
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://example.com",
                    "body": small_body,
                },
            }

            output = await handler.execute(envelope)
            result = output.result
            assert result["status"] == "success"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_response_within_limit_succeeds(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that response within size limit succeeds."""
        config: dict[str, object] = {"max_response_size": 1000}  # 1000 bytes
        await handler.initialize(config)

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "text/plain"},
            body_bytes=b"x" * 50,  # 50 bytes, under 1000 byte limit
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            output = await handler.execute(envelope)
            result = output.result
            assert result["status"] == "success"
            assert result["payload"]["body"] == "x" * 50

        await handler.shutdown()

    def test_size_limits_in_describe(self, handler: HttpRestHandler) -> None:
        """Test that size limits are included in describe response."""
        description = handler.describe()

        # Default values
        assert description["max_request_size"] == 10 * 1024 * 1024
        assert description["max_response_size"] == 50 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_invalid_config_uses_defaults(self, handler: HttpRestHandler) -> None:
        """Test that invalid config values use defaults."""
        config: dict[str, object] = {
            "max_request_size": -100,  # Invalid negative
            "max_response_size": "not a number",  # Invalid type
        }
        await handler.initialize(config)

        # Should use defaults for invalid values
        assert handler._max_request_size == 10 * 1024 * 1024
        assert handler._max_response_size == 50 * 1024 * 1024

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_content_length_header_validation_rejects_large_response(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that Content-Length header is validated BEFORE reading response body.

        This is the critical security fix - responses with Content-Length > limit
        should be rejected immediately without reading the body into memory.
        """
        config: dict[str, object] = {"max_response_size": 100}  # 100 bytes
        await handler.initialize(config)

        # Create response with Content-Length header indicating size > limit
        # The actual body doesn't matter because we should reject based on header
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={
                "content-type": "text/plain",
                "content-length": "1000000",  # 1 MB, way over 100 byte limit
            },
            body_bytes=b"x" * 10,  # Small actual body - shouldn't matter
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/large-file"},
            }

            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            error_msg = str(exc_info.value)
            # Should mention Content-Length in the error with sanitized size category
            assert "Content-Length" in error_msg
            assert "exceeds configured limit" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_content_length_header_validation_allows_small_response(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that Content-Length header validation allows responses under limit."""
        config: dict[str, object] = {"max_response_size": 1000}  # 1000 bytes
        await handler.initialize(config)

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={
                "content-type": "text/plain",
                "content-length": "50",  # 50 bytes, under limit
            },
            body_bytes=b"x" * 50,
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/small-file"},
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["status"] == "success"
            assert result["payload"]["body"] == "x" * 50

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_validation_for_chunked_responses(
        self, handler: HttpRestHandler
    ) -> None:
        """Test that responses without Content-Length are validated during streaming.

        Chunked transfer encoding and other cases without Content-Length header
        should be validated as the body is read, stopping early if limit exceeded.
        """
        config: dict[str, object] = {"max_response_size": 50}  # 50 bytes
        await handler.initialize(config)

        # No Content-Length header (simulating chunked transfer)
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "text/plain"},  # No content-length
            body_bytes=b"x" * 100,  # 100 bytes, exceeds 50 byte limit
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/chunked-data"},
            }

            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            error_msg = str(exc_info.value)
            # Should mention streaming read in the error with sanitized size category
            assert "streaming read" in error_msg
            assert "exceeds configured limit" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_content_length_zero_succeeds(self, handler: HttpRestHandler) -> None:
        """Test Content-Length: 0 returns empty response successfully.

        Content-Length of 0 is a valid HTTP response indicating an empty body.
        This should succeed without errors.
        """
        config: dict[str, object] = {"max_response_size": 100}  # 100 bytes
        await handler.initialize(config)

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "content-length": "0",
            },
            body_bytes=b"",  # Empty body
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/empty-response"},
            }

            output = await handler.execute(envelope)
            result = output.result

            assert result["status"] == "success"
            # Empty body parsed as empty string (not JSON because empty string is invalid JSON)
            assert result["payload"]["body"] == ""
            assert result["payload"]["status_code"] == 200

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_content_length_with_whitespace_handled(
        self, handler: HttpRestHandler
    ) -> None:
        """Test Content-Length with leading/trailing whitespace parses correctly.

        HTTP headers may have whitespace around values. Python's int() function
        automatically strips leading/trailing whitespace, so " 50 " is parsed as 50.
        This test verifies that whitespace in Content-Length headers doesn't cause
        any issues and the size validation works correctly.
        """
        config: dict[str, object] = {"max_response_size": 100}  # 100 bytes
        await handler.initialize(config)

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={
                "content-type": "text/plain",
                "content-length": " 50 ",  # Whitespace around value
            },
            body_bytes=b"x" * 50,  # 50 bytes - within limit
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/whitespace-header"},
            }

            # Should succeed - either by parsing " 50 " as 50 or falling through
            # to streaming validation (which passes for 50 bytes)
            output = await handler.execute(envelope)
            result = output.result

            assert result["status"] == "success"
            assert result["payload"]["body"] == "x" * 50
            assert result["payload"]["status_code"] == 200

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_content_length_headers_handled(
        self, handler: HttpRestHandler
    ) -> None:
        """Test multiple Content-Length headers are handled gracefully.

        When multiple Content-Length values exist (undefined behavior per HTTP spec),
        httpx typically returns them comma-separated or uses the first value.
        The handler should not crash regardless of the format.
        """
        config: dict[str, object] = {"max_response_size": 100}  # 100 bytes
        await handler.initialize(config)

        # httpx may represent multiple headers as comma-separated values
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={
                "content-type": "text/plain",
                "content-length": "30, 30",  # Multiple values (invalid per strict HTTP)
            },
            body_bytes=b"x" * 30,  # 30 bytes - within limit
            content_type="text/plain",
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/multiple-content-length"},
            }

            # Should succeed - the invalid "30, 30" cannot be parsed as int
            # so it falls through to streaming validation which passes for 30 bytes
            output = await handler.execute(envelope)
            result = output.result

            assert result["status"] == "success"
            assert result["payload"]["body"] == "x" * 30
            assert result["payload"]["status_code"] == 200

        await handler.shutdown()


class TestHttpRestHandlerLogWarnings:
    """Test suite for log warning assertions (OMN-252 acceptance criteria).

    These tests verify that:
    1. Normal operations produce no unexpected warnings
    2. Expected warnings are logged only in specific error conditions
    """

    HANDLER_MODULE = "omnibase_infra.handlers.handler_http"

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_no_unexpected_warnings_during_normal_operation(
        self, handler: HttpRestHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that normal operations produce no unexpected warnings.

        This test verifies the OMN-252 acceptance criteria: "Asserts no unexpected
        warnings in logs" during normal handler lifecycle and execution.
        """
        # Create mock response for GET operation
        import json as json_module
        import logging

        body_data = {"data": "test_value"}
        body_bytes = json_module.dumps(body_data).encode("utf-8")
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=body_bytes,
        )

        with caplog.at_level(logging.WARNING):
            # Initialize with valid config
            await handler.initialize({})

            # Perform normal GET operation
            with patch.object(handler._client, "stream") as mock_stream:
                mock_stream.return_value = mock_stream_context(mock_response)

                correlation_id = uuid4()
                envelope: dict[str, object] = {
                    "operation": "http.get",
                    "payload": {"url": "https://api.example.com/resource"},
                    "correlation_id": correlation_id,
                }

                output = await handler.execute(envelope)
                result = output.result
                assert result["status"] == "success"

            # Shutdown
            await handler.shutdown()

        # Filter for warnings from our handler module
        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 0, (
            f"Unexpected warnings: {[w.message for w in handler_warnings]}"
        )

    @pytest.mark.asyncio
    async def test_expected_warning_on_invalid_max_request_size(
        self, handler: HttpRestHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid max_request_size config produces expected warning.

        When an invalid max_request_size is provided (e.g., negative number or
        wrong type), the handler should log a warning and use the default value.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            # Initialize with invalid max_request_size (negative value)
            config: dict[str, object] = {"max_request_size": -100}
            await handler.initialize(config)

            # Verify default was used
            assert handler._max_request_size == 10 * 1024 * 1024

            await handler.shutdown()

        # Should have exactly one warning about invalid config
        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 1
        assert "Invalid max_request_size" in handler_warnings[0].message

    @pytest.mark.asyncio
    async def test_expected_warning_on_invalid_max_response_size(
        self, handler: HttpRestHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid max_response_size config produces expected warning.

        When an invalid max_response_size is provided (e.g., string instead of int),
        the handler should log a warning and use the default value.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            # Initialize with invalid max_response_size (wrong type)
            config: dict[str, object] = {"max_response_size": "not a number"}
            await handler.initialize(config)

            # Verify default was used
            assert handler._max_response_size == 50 * 1024 * 1024

            await handler.shutdown()

        # Should have exactly one warning about invalid config
        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 1
        assert "Invalid max_response_size" in handler_warnings[0].message

    @pytest.mark.asyncio
    async def test_expected_warning_on_invalid_content_length_header(
        self, handler: HttpRestHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid Content-Length header produces expected warning.

        When a response has an invalid Content-Length header (e.g., non-numeric),
        the handler should log a warning and fall back to streaming validation.
        """
        import logging

        config: dict[str, object] = {"max_response_size": 1000}
        await handler.initialize(config)

        # Create response with invalid Content-Length header
        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={
                "content-type": "text/plain",
                "content-length": "invalid-number",
            },
            body_bytes=b"x" * 50,
            content_type="text/plain",
        )

        with caplog.at_level(logging.WARNING):
            with patch.object(handler._client, "stream") as mock_stream:
                mock_stream.return_value = mock_stream_context(mock_response)

                envelope: dict[str, object] = {
                    "operation": "http.get",
                    "payload": {"url": "https://example.com/resource"},
                }

                output = await handler.execute(envelope)
                result = output.result
                assert result["status"] == "success"

        await handler.shutdown()

        # Should have a warning about invalid Content-Length header
        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 1
        assert "Invalid Content-Length" in handler_warnings[0].message


class TestHttpRestHandlerPrepareRequestContent:
    """Test suite for _prepare_request_content helper method."""

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.fixture
    def error_context(self) -> MagicMock:
        """Create mock error context."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import ModelInfraErrorContext

        return ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="http.post",
            target_name="https://example.com",
            correlation_id=uuid4(),
        )

    def test_get_method_returns_empty_content(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test GET method returns empty content regardless of body."""
        content, json_body, headers = handler._prepare_request_content(
            method="GET",
            headers={"X-Custom": "value"},
            body={"data": "ignored"},
            pre_serialized=None,
            ctx=error_context,
        )

        assert content is None
        assert json_body is None
        assert headers == {"X-Custom": "value"}

    def test_post_with_none_body_returns_empty_content(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test POST with None body returns empty content."""
        content, json_body, headers = handler._prepare_request_content(
            method="POST",
            headers={},
            body=None,
            pre_serialized=None,
            ctx=error_context,
        )

        assert content is None
        assert json_body is None
        assert headers == {}

    def test_post_with_pre_serialized_bytes(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test POST with pre-serialized bytes uses them directly."""
        pre_serialized = b'{"key": "value"}'

        content, json_body, headers = handler._prepare_request_content(
            method="POST",
            headers={},
            body={"key": "value"},  # Body is ignored when pre_serialized is provided
            pre_serialized=pre_serialized,
            ctx=error_context,
        )

        assert content == pre_serialized
        assert json_body is None
        assert headers == {"Content-Type": "application/json"}

    def test_post_with_pre_serialized_preserves_existing_content_type(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test POST with pre-serialized preserves existing Content-Type header."""
        pre_serialized = b'{"key": "value"}'

        content, json_body, headers = handler._prepare_request_content(
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
            body={"key": "value"},
            pre_serialized=pre_serialized,
            ctx=error_context,
        )

        assert content == pre_serialized
        assert json_body is None
        # Original Content-Type preserved (case-insensitive check)
        assert headers == {"Content-Type": "application/json; charset=utf-8"}

    def test_post_with_dict_body_fallback(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test POST with dict body uses json parameter when no pre-serialized."""
        body = {"key": "value", "number": 42}

        content, json_body, headers = handler._prepare_request_content(
            method="POST",
            headers={},
            body=body,
            pre_serialized=None,
            ctx=error_context,
        )

        assert content is None
        assert json_body == body
        assert headers == {}

    def test_post_with_string_body(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test POST with string body uses content parameter."""
        body = "raw string content"

        content, json_body, headers = handler._prepare_request_content(
            method="POST",
            headers={},
            body=body,
            pre_serialized=None,
            ctx=error_context,
        )

        assert content == "raw string content"
        assert json_body is None
        assert headers == {}

    def test_post_with_list_body_serializes_to_json(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test POST with list body serializes to JSON string."""
        body = [1, 2, 3, "four"]

        content, json_body, headers = handler._prepare_request_content(
            method="POST",
            headers={},
            body=body,
            pre_serialized=None,
            ctx=error_context,
        )

        assert content == '[1, 2, 3, "four"]'
        assert json_body is None
        assert headers == {}

    def test_post_with_non_serializable_body_raises_error(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test POST with non-JSON-serializable body raises ProtocolConfigurationError."""

        class NonSerializable:
            pass

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            handler._prepare_request_content(
                method="POST",
                headers={},
                body=NonSerializable(),
                pre_serialized=None,
                ctx=error_context,
            )

        assert "not JSON-serializable" in str(exc_info.value)
        assert "NonSerializable" in str(exc_info.value)

    def test_headers_not_mutated(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test original headers dict is not mutated."""
        original_headers = {"X-Custom": "value"}
        pre_serialized = b'{"key": "value"}'

        _content, _json_body, headers = handler._prepare_request_content(
            method="POST",
            headers=original_headers,
            body={"key": "value"},
            pre_serialized=pre_serialized,
            ctx=error_context,
        )

        # Original headers should be unchanged
        assert original_headers == {"X-Custom": "value"}
        # Returned headers should have Content-Type added
        assert headers == {"X-Custom": "value", "Content-Type": "application/json"}

    def test_content_type_check_case_insensitive(
        self, handler: HttpRestHandler, error_context: MagicMock
    ) -> None:
        """Test Content-Type header check is case-insensitive."""
        pre_serialized = b'{"key": "value"}'

        # Lowercase content-type
        _content, _json_body, headers = handler._prepare_request_content(
            method="POST",
            headers={"content-type": "text/plain"},
            body={"key": "value"},
            pre_serialized=pre_serialized,
            ctx=error_context,
        )

        # Should not add another Content-Type since one exists (case-insensitive)
        assert headers == {"content-type": "text/plain"}
        assert "Content-Type" not in headers


class TestHttpRestHandlerDeterministicIntegration:
    """Integration tests demonstrating deterministic test utilities (OMN-252).

    These tests validate that the deterministic utilities from
    tests.helpers.deterministic work correctly in handler tests.
    """

    @pytest.fixture
    def handler(self) -> HttpRestHandler:
        """Create HttpRestHandler fixture."""
        return HttpRestHandler()

    @pytest.mark.asyncio
    async def test_deterministic_correlation_id_in_full_flow(
        self, handler: HttpRestHandler
    ) -> None:
        """Test full HTTP flow with deterministic correlation ID.

        Demonstrates DeterministicIdGenerator providing predictable UUIDs
        for reproducible test assertions.
        """
        await handler.initialize({})

        # Create deterministic ID generator with known seed
        id_gen = DeterministicIdGenerator(seed=1000)

        # Generate predictable correlation ID
        correlation_id = id_gen.next_uuid()

        # Verify it's deterministic (seed=1000, next_uuid returns UUID(int=1001))
        assert correlation_id.int == 1001

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b'{"result": "ok"}',
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/test"},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result

            # Verify deterministic correlation ID flows through (as string in result)
            assert result["correlation_id"] == str(correlation_id)
            assert result["status"] == "success"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_deterministic_clock_for_timing_assertions(
        self, handler: HttpRestHandler
    ) -> None:
        """Test DeterministicClock provides controllable time for timing tests.

        Demonstrates DeterministicClock enabling predictable time-based
        assertions in handler tests without relying on real wall-clock time.
        """
        await handler.initialize({})

        # Create deterministic clock at a known start time
        clock = DeterministicClock()
        start_time = clock.now()

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b'{"timestamp": "2025-01-01T00:00:00Z"}',
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/time"},
            }

            output = await handler.execute(envelope)
            result = output.result
            assert result["status"] == "success"

        # Advance clock to simulate elapsed time
        clock.advance(60)  # 60 seconds
        end_time = clock.now()

        # Verify deterministic time advancement
        elapsed = (end_time - start_time).total_seconds()
        assert elapsed == 60.0, "Clock should advance exactly 60 seconds"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_deterministic_utilities_combined_in_handler_flow(
        self, handler: HttpRestHandler
    ) -> None:
        """Test both deterministic utilities working together in full flow.

        Demonstrates DeterministicIdGenerator and DeterministicClock used
        together for comprehensive reproducible test assertions.
        """
        await handler.initialize({})

        # Initialize both deterministic utilities
        id_gen = DeterministicIdGenerator(seed=500)
        clock = DeterministicClock()

        # Generate multiple predictable correlation IDs
        correlation_id_1 = id_gen.next_uuid()
        correlation_id_2 = id_gen.next_uuid()

        # Verify sequential determinism
        assert correlation_id_1.int == 501
        assert correlation_id_2.int == 502

        # Record start time
        request_start = clock.now()

        mock_response = create_mock_streaming_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body_bytes=b'{"data": "test"}',
        )

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            # First request with first correlation ID
            envelope_1: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/resource/1"},
                "correlation_id": correlation_id_1,
            }

            output_1 = await handler.execute(envelope_1)
            result_1 = output_1.result
            assert result_1["correlation_id"] == str(correlation_id_1)

        # Simulate 30 second delay between requests
        clock.advance(30)

        with patch.object(handler._client, "stream") as mock_stream:
            mock_stream.return_value = mock_stream_context(mock_response)

            # Second request with second correlation ID
            envelope_2: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/resource/2"},
                "correlation_id": correlation_id_2,
            }

            output_2 = await handler.execute(envelope_2)
            result_2 = output_2.result
            assert result_2["correlation_id"] == str(correlation_id_2)

        # Record end time and verify deterministic timing
        request_end = clock.now()
        total_elapsed = (request_end - request_start).total_seconds()
        assert total_elapsed == 30.0, "Total elapsed time should be exactly 30 seconds"

        # Verify ID generator state is predictable
        assert id_gen.current_counter == 502

        await handler.shutdown()


__all__: list[str] = [
    "TestHttpRestHandlerInitialization",
    "TestHttpRestHandlerGetOperations",
    "TestHttpRestHandlerPostOperations",
    "TestHttpRestHandlerErrorHandling",
    "TestHttpRestHandlerDescribe",
    "TestHttpRestHandlerLifecycle",
    "TestHttpRestHandlerCorrelationId",
    "TestHttpRestHandlerResponseParsing",
    "TestHttpRestHandlerSizeLimits",
    "TestHttpRestHandlerLogWarnings",
    "TestHttpRestHandlerPrepareRequestContent",
    "TestHttpRestHandlerDeterministicIntegration",
]

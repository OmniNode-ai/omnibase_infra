# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HttpRestAdapter.

Comprehensive test suite covering initialization, GET/POST operations,
error handling, health checks, describe, and lifecycle management.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from omnibase_core.enums.enum_handler_type import EnumHandlerType

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_http import HttpRestAdapter

# Type alias for response dict with nested structure
ResponseDict = dict[str, object]


class TestHttpRestAdapterInitialization:
    """Test suite for HttpRestAdapter initialization."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    def test_handler_init_default_state(self, handler: HttpRestAdapter) -> None:
        """Test handler initializes in uninitialized state."""
        assert handler._initialized is False
        assert handler._client is None
        assert handler._timeout == 30.0

    def test_handler_type_returns_http(self, handler: HttpRestAdapter) -> None:
        """Test handler_type property returns EnumHandlerType.HTTP."""
        assert handler.handler_type == EnumHandlerType.HTTP

    @pytest.mark.asyncio
    async def test_initialize_with_empty_config(self, handler: HttpRestAdapter) -> None:
        """Test handler initializes with empty config (uses defaults)."""
        await handler.initialize({})

        assert handler._initialized is True
        assert handler._client is not None
        assert handler._timeout == 30.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_config_dict(self, handler: HttpRestAdapter) -> None:
        """Test handler initializes with config dict (config ignored in MVP)."""
        config: dict[str, object] = {"timeout": 60.0, "custom_option": "value"}
        await handler.initialize(config)

        # MVP ignores config, uses fixed 30s timeout
        assert handler._initialized is True
        assert handler._timeout == 30.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_creates_async_client(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test initialize creates httpx.AsyncClient with correct timeout."""
        await handler.initialize({})

        assert isinstance(handler._client, httpx.AsyncClient)
        # Verify timeout is set (30s default)
        assert handler._client.timeout.connect == 30.0

        await handler.shutdown()


class TestHttpRestAdapterGetOperations:
    """Test suite for HTTP GET operations."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    @pytest.mark.asyncio
    async def test_get_successful_response(self, handler: HttpRestAdapter) -> None:
        """Test successful GET request returns correct response structure."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",
            "x-custom": "value",
        }
        mock_response.json.return_value = {"data": "test_value"}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/resource"},
                "correlation_id": str(uuid4()),
            }

            result = cast(ResponseDict, await handler.execute(envelope))

            assert result["status"] == "success"
            payload = cast(ResponseDict, result["payload"])
            assert payload["status_code"] == 200
            assert payload["body"] == {"data": "test_value"}
            assert "correlation_id" in result

            mock_get.assert_called_once_with(
                "https://api.example.com/resource", headers={}
            )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_with_custom_headers(self, handler: HttpRestAdapter) -> None:
        """Test GET request passes custom headers correctly."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "OK"
        mock_response.json.side_effect = Exception("Not JSON")

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

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

            result = await handler.execute(envelope)

            mock_get.assert_called_once_with(
                "https://api.example.com/status",
                headers={"Authorization": "Bearer token123", "X-Request-ID": "req-456"},
            )

            assert result["payload"]["body"] == "OK"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_with_query_params_in_url(self, handler: HttpRestAdapter) -> None:
        """Test GET request with query parameters in URL."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"items": [1, 2, 3]}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {
                    "url": "https://api.example.com/items?page=1&limit=10&filter=active",
                },
            }

            result = await handler.execute(envelope)

            mock_get.assert_called_once_with(
                "https://api.example.com/items?page=1&limit=10&filter=active",
                headers={},
            )

            assert result["payload"]["body"] == {"items": [1, 2, 3]}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_text_response(self, handler: HttpRestAdapter) -> None:
        """Test GET request with text/plain response."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain; charset=utf-8"}
        mock_response.text = "Hello, World!"

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com/hello"},
            }

            result = await handler.execute(envelope)

            assert result["payload"]["body"] == "Hello, World!"
            assert result["payload"]["status_code"] == 200

        await handler.shutdown()


class TestHttpRestAdapterPostOperations:
    """Test suite for HTTP POST operations."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    @pytest.mark.asyncio
    async def test_post_with_json_body(self, handler: HttpRestAdapter) -> None:
        """Test POST request with JSON body."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"id": 123, "created": True}

        with patch.object(handler._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/users",
                    "body": {"name": "John", "email": "john@example.com"},
                },
            }

            result = await handler.execute(envelope)

            mock_post.assert_called_once_with(
                "https://api.example.com/users",
                headers={},
                json={"name": "John", "email": "john@example.com"},
            )

            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 201
            assert result["payload"]["body"] == {"id": 123, "created": True}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_string_body(self, handler: HttpRestAdapter) -> None:
        """Test POST request with string body."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Received"

        with patch.object(handler._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/message",
                    "body": "Hello from client",
                },
            }

            result = await handler.execute(envelope)

            mock_post.assert_called_once_with(
                "https://api.example.com/message",
                headers={},
                content="Hello from client",
            )

            assert result["payload"]["body"] == "Received"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_no_body(self, handler: HttpRestAdapter) -> None:
        """Test POST request with no body."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 204
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = ""

        with patch.object(handler._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {"url": "https://api.example.com/trigger"},
            }

            result = await handler.execute(envelope)

            mock_post.assert_called_once_with(
                "https://api.example.com/trigger",
                headers={},
            )

            assert result["payload"]["status_code"] == 204

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_custom_headers(self, handler: HttpRestAdapter) -> None:
        """Test POST request with custom headers."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"success": True}

        with patch.object(handler._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/data",
                    "headers": {
                        "Content-Type": "application/json",
                        "X-API-Key": "secret-key-123",
                    },
                    "body": {"value": 42},
                },
            }

            result = await handler.execute(envelope)

            mock_post.assert_called_once_with(
                "https://api.example.com/data",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": "secret-key-123",
                },
                json={"value": 42},
            )

            assert result["payload"]["body"] == {"success": True}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_post_with_list_body_serialized_to_json(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test POST with list body gets JSON serialized."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"processed": 3}

        with patch.object(handler._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/batch",
                    "body": [1, 2, 3],  # List body, not dict or string
                },
            }

            result = await handler.execute(envelope)

            # List body uses content= with json.dumps()
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args.kwargs["content"] == "[1, 2, 3]"

            assert result["payload"]["body"] == {"processed": 3}

        await handler.shutdown()


class TestHttpRestAdapterErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    @pytest.mark.asyncio
    async def test_timeout_error_raises_infra_timeout(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test timeout error is converted to InfraTimeoutError."""
        await handler.initialize({})

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timed out")

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
        self, handler: HttpRestAdapter
    ) -> None:
        """Test connection error is converted to InfraConnectionError."""
        await handler.initialize({})

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

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
        self, handler: HttpRestAdapter
    ) -> None:
        """Test http.put operation raises RuntimeHostError (not supported in MVP)."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.put",
            "payload": {"url": "https://api.example.com/resource/123"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "http.put" in str(exc_info.value)
        assert "not supported" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unsupported_operation_delete_raises_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test http.delete operation raises RuntimeHostError (not supported in MVP)."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.delete",
            "payload": {"url": "https://api.example.com/resource/123"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "http.delete" in str(exc_info.value)
        assert "not supported" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unsupported_operation_patch_raises_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test http.patch operation raises RuntimeHostError (not supported in MVP)."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.patch",
            "payload": {"url": "https://api.example.com/resource/123"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "http.patch" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_url_field_raises_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test missing URL field raises RuntimeHostError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"headers": {"X-Test": "value"}},  # No URL
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "url" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_empty_url_field_raises_error(self, handler: HttpRestAdapter) -> None:
        """Test empty URL field raises RuntimeHostError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": ""},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "url" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_operation_raises_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test missing operation field raises RuntimeHostError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "payload": {"url": "https://example.com"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "operation" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_payload_raises_error(self, handler: HttpRestAdapter) -> None:
        """Test missing payload field raises RuntimeHostError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "payload" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_headers_type_raises_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test invalid headers type raises RuntimeHostError."""
        await handler.initialize({})

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {
                "url": "https://example.com",
                "headers": "not-a-dict",  # Invalid type
            },
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "headers" in str(exc_info.value).lower()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_http_status_error_returns_response(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test HTTPStatusError still returns the response (not an exception)."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": "Not found"}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            http_error = httpx.HTTPStatusError(
                "404 Not Found", request=MagicMock(), response=mock_response
            )
            mock_get.side_effect = http_error

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/missing"},
            }

            result = await handler.execute(envelope)

            # Should return the error response, not raise
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 404
            assert result["payload"]["body"] == {"error": "Not found"}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_generic_http_error_raises_connection_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test generic HTTPError raises InfraConnectionError."""
        await handler.initialize({})

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Unknown HTTP error")

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://api.example.com/broken"},
            }

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(envelope)

            assert "HTTP error" in str(exc_info.value)

        await handler.shutdown()


class TestHttpRestAdapterHealthCheck:
    """Test suite for health check operations."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    @pytest.mark.asyncio
    async def test_health_check_structure(self, handler: HttpRestAdapter) -> None:
        """Test health_check returns correct structure."""
        await handler.initialize({})

        health = await handler.health_check()

        assert "healthy" in health
        assert "initialized" in health
        assert "adapter_type" in health
        assert "timeout_seconds" in health

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_initialized(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test health_check shows healthy=True when initialized."""
        await handler.initialize({})

        health = await handler.health_check()

        assert health["healthy"] is True
        assert health["initialized"] is True
        assert health["adapter_type"] == "http"
        assert health["timeout_seconds"] == 30.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_not_initialized(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test health_check shows healthy=False when not initialized."""
        health = await handler.health_check()

        assert health["healthy"] is False
        assert health["initialized"] is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_after_shutdown(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test health_check shows healthy=False after shutdown."""
        await handler.initialize({})
        await handler.shutdown()

        health = await handler.health_check()

        assert health["healthy"] is False
        assert health["initialized"] is False


class TestHttpRestAdapterDescribe:
    """Test suite for describe operations."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    def test_describe_returns_handler_metadata(self, handler: HttpRestAdapter) -> None:
        """Test describe returns correct handler metadata."""
        description = handler.describe()

        assert description["adapter_type"] == "http"
        assert description["timeout_seconds"] == 30.0
        assert description["version"] == "0.1.0-mvp"
        assert description["initialized"] is False

    def test_describe_lists_supported_operations(
        self, handler: HttpRestAdapter
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
        self, handler: HttpRestAdapter
    ) -> None:
        """Test describe shows correct initialized state."""
        assert handler.describe()["initialized"] is False

        await handler.initialize({})
        assert handler.describe()["initialized"] is True

        await handler.shutdown()
        assert handler.describe()["initialized"] is False


class TestHttpRestAdapterLifecycle:
    """Test suite for lifecycle management."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self, handler: HttpRestAdapter) -> None:
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
        self, handler: HttpRestAdapter
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
        self, handler: HttpRestAdapter
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
    async def test_multiple_shutdown_calls_safe(self, handler: HttpRestAdapter) -> None:
        """Test multiple shutdown calls are safe (idempotent)."""
        await handler.initialize({})
        await handler.shutdown()
        await handler.shutdown()  # Second call should not raise

        assert handler._initialized is False
        assert handler._client is None

    @pytest.mark.asyncio
    async def test_reinitialize_after_shutdown(self, handler: HttpRestAdapter) -> None:
        """Test handler can be reinitialized after shutdown."""
        await handler.initialize({})
        await handler.shutdown()

        assert handler._initialized is False

        await handler.initialize({})

        assert handler._initialized is True
        assert handler._client is not None

        await handler.shutdown()


class TestHttpRestAdapterCorrelationId:
    """Test suite for correlation ID handling."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_uuid(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test correlation ID extracted from envelope as UUID."""
        await handler.initialize({})

        correlation_id = uuid4()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
                "correlation_id": correlation_id,
            }

            result = await handler.execute(envelope)

            assert result["correlation_id"] == str(correlation_id)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_string(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test correlation ID extracted from envelope as string."""
        await handler.initialize({})

        correlation_id = str(uuid4())
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
                "correlation_id": correlation_id,
            }

            result = await handler.execute(envelope)

            assert result["correlation_id"] == correlation_id

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_generated_when_missing(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test correlation ID generated when not in envelope."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            result = await handler.execute(envelope)

            # Should have a generated UUID
            assert "correlation_id" in result
            # Verify it's a valid UUID string
            UUID(result["correlation_id"])

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_invalid_string_generates_new(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test invalid correlation ID string generates new UUID."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
                "correlation_id": "not-a-valid-uuid",
            }

            result = await handler.execute(envelope)

            # Should have a generated UUID (not the invalid string)
            assert "correlation_id" in result
            generated_id = result["correlation_id"]
            assert generated_id != "not-a-valid-uuid"
            # Verify it's a valid UUID string
            UUID(generated_id)

        await handler.shutdown()


class TestHttpRestAdapterResponseParsing:
    """Test suite for response parsing."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    @pytest.mark.asyncio
    async def test_json_response_parsed(self, handler: HttpRestAdapter) -> None:
        """Test JSON response is parsed correctly."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json; charset=utf-8"}
        mock_response.json.return_value = {"key": "value", "nested": {"a": 1}}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            result = await handler.execute(envelope)

            assert result["payload"]["body"] == {"key": "value", "nested": {"a": 1}}

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_text(self, handler: HttpRestAdapter) -> None:
        """Test invalid JSON response falls back to text."""
        import json as json_module

        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.side_effect = json_module.JSONDecodeError(
            "Invalid JSON", "doc", 0
        )
        mock_response.text = "Not valid JSON {"

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            result = await handler.execute(envelope)

            assert result["payload"]["body"] == "Not valid JSON {"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_non_json_content_type_returns_text(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test non-JSON content type returns text body."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = "<html><body>Hello</body></html>"

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            result = await handler.execute(envelope)

            assert result["payload"]["body"] == "<html><body>Hello</body></html>"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_response_headers_included(self, handler: HttpRestAdapter) -> None:
        """Test response headers are included in result."""
        await handler.initialize({})

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",
            "x-request-id": "req-123",
            "x-rate-limit-remaining": "99",
        }
        mock_response.json.return_value = {}

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            result = await handler.execute(envelope)

            assert result["payload"]["headers"]["content-type"] == "application/json"
            assert result["payload"]["headers"]["x-request-id"] == "req-123"
            assert result["payload"]["headers"]["x-rate-limit-remaining"] == "99"

        await handler.shutdown()


class TestHttpRestAdapterSizeLimits:
    """Test suite for request/response size limits."""

    @pytest.fixture
    def handler(self) -> HttpRestAdapter:
        """Create HttpRestAdapter fixture."""
        return HttpRestAdapter()

    def test_default_size_limits(self, handler: HttpRestAdapter) -> None:
        """Test default size limits are set correctly."""
        assert handler._max_request_size == 10 * 1024 * 1024  # 10 MB
        assert handler._max_response_size == 50 * 1024 * 1024  # 50 MB

    @pytest.mark.asyncio
    async def test_configurable_size_limits(self, handler: HttpRestAdapter) -> None:
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
        self, handler: HttpRestAdapter
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

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "exceeds limit" in str(exc_info.value)
        assert "10 bytes" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_size_validation_dict_body(
        self, handler: HttpRestAdapter
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

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "exceeds limit" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_size_validation_bytes_body(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test request size validation with bytes body."""
        config: dict[str, object] = {"max_request_size": 5}  # 5 bytes
        await handler.initialize(config)

        # Create a test that uses bytes body via list serialization
        # Since bytes can't be passed directly through payload.get("body"),
        # we test the internal method directly
        correlation_id = uuid4()

        # Test the internal validation method with bytes
        with pytest.raises(RuntimeHostError) as exc_info:
            handler._validate_request_size(b"123456", correlation_id)

        assert "exceeds limit" in str(exc_info.value)
        assert "5 bytes" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_size_exceeds_limit_raises_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test that request exceeding size limit raises RuntimeHostError."""
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

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        error_msg = str(exc_info.value)
        assert "200 bytes" in error_msg
        assert "100 bytes" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_response_size_exceeds_limit_raises_error(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test that response exceeding size limit raises RuntimeHostError."""
        config: dict[str, object] = {"max_response_size": 50}  # 50 bytes
        await handler.initialize(config)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "x" * 100  # 100 bytes, exceeds 50 byte limit

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            error_msg = str(exc_info.value)
            assert "exceeds limit" in error_msg
            assert "50 bytes" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_none_body_skips_validation(self, handler: HttpRestAdapter) -> None:
        """Test that None body skips size validation."""
        config: dict[str, object] = {"max_request_size": 1}  # 1 byte - very small
        await handler.initialize(config)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 204
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = ""

        with patch.object(handler._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # POST with no body should not raise size limit error
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {"url": "https://example.com"},  # No body
            }

            result = await handler.execute(envelope)
            assert result["status"] == "success"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_request_within_limit_succeeds(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test that request within size limit succeeds."""
        config: dict[str, object] = {"max_request_size": 1000}  # 1000 bytes
        await handler.initialize(config)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"success": True}

        with patch.object(handler._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            small_body = "x" * 50  # 50 bytes, under 1000 byte limit
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": "https://example.com",
                    "body": small_body,
                },
            }

            result = await handler.execute(envelope)
            assert result["status"] == "success"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_response_within_limit_succeeds(
        self, handler: HttpRestAdapter
    ) -> None:
        """Test that response within size limit succeeds."""
        config: dict[str, object] = {"max_response_size": 1000}  # 1000 bytes
        await handler.initialize(config)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "x" * 50  # 50 bytes, under 1000 byte limit

        with patch.object(handler._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            }

            result = await handler.execute(envelope)
            assert result["status"] == "success"
            assert result["payload"]["body"] == "x" * 50

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_size_limits_in_health_check(self, handler: HttpRestAdapter) -> None:
        """Test that size limits are included in health check response."""
        config: dict[str, object] = {
            "max_request_size": 5000,
            "max_response_size": 10000,
        }
        await handler.initialize(config)

        health = await handler.health_check()

        assert health["max_request_size"] == 5000
        assert health["max_response_size"] == 10000

        await handler.shutdown()

    def test_size_limits_in_describe(self, handler: HttpRestAdapter) -> None:
        """Test that size limits are included in describe response."""
        description = handler.describe()

        # Default values
        assert description["max_request_size"] == 10 * 1024 * 1024
        assert description["max_response_size"] == 50 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_invalid_config_uses_defaults(self, handler: HttpRestAdapter) -> None:
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


__all__: list[str] = [
    "TestHttpRestAdapterInitialization",
    "TestHttpRestAdapterGetOperations",
    "TestHttpRestAdapterPostOperations",
    "TestHttpRestAdapterErrorHandling",
    "TestHttpRestAdapterHealthCheck",
    "TestHttpRestAdapterDescribe",
    "TestHttpRestAdapterLifecycle",
    "TestHttpRestAdapterCorrelationId",
    "TestHttpRestAdapterResponseParsing",
    "TestHttpRestAdapterSizeLimits",
]

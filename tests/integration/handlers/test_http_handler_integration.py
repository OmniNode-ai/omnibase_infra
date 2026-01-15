# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerHttpRest using pytest-httpserver.

These tests verify HTTP handler behavior against a real local HTTP server,
testing actual network communication without relying on external services.

All tests are marked with @pytest.mark.integration and are designed to be
run locally (not in CI) to validate HTTP protocol handling.

Requirements:
    pytest-httpserver must be installed: pip install pytest-httpserver

Test Coverage:
    - HTTP GET operations (success, with headers)
    - HTTP POST operations (JSON body, with headers)
    - Error responses (404, 500)
    - Timeout handling
    - Response size limit enforcement
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

# Skip entire module if pytest-httpserver is not installed (required for HTTP integration tests)
pytest.importorskip("pytest_httpserver")

# werkzeug is a transitive dependency of pytest-httpserver, used for Response handling
from werkzeug import Response

from omnibase_infra.errors import InfraTimeoutError, InfraUnavailableError
from omnibase_infra.handlers import HandlerHttpRest

if TYPE_CHECKING:
    from pytest_httpserver import HTTPServer


pytestmark = [pytest.mark.asyncio]


class TestHttpGetSuccess:
    """Integration tests for successful HTTP GET operations."""

    async def test_http_get_success(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test successful HTTP GET request returns correct response structure.

        Verifies that:
        - GET request is properly sent to the mock server
        - Response status code is captured correctly
        - JSON response body is parsed correctly
        - Response headers are included
        """
        # Arrange - Configure mock server endpoint
        expected_response = {"message": "Hello, World!", "count": 42}
        httpserver.expect_request("/api/resource").respond_with_json(expected_response)

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute GET request
            correlation_id = uuid4()
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/resource")},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify response structure and content
            assert result["status"] == "success"
            payload = result["payload"]
            assert payload["status_code"] == 200
            assert payload["body"] == expected_response
            assert "content-type" in payload["headers"]
            assert "application/json" in payload["headers"]["content-type"]
            assert result["correlation_id"] == str(correlation_id)

        finally:
            await handler.shutdown()


class TestHttpGetWithHeaders:
    """Integration tests for HTTP GET operations with custom headers."""

    async def test_http_get_with_headers(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP GET request passes custom headers to server.

        Verifies that:
        - Custom request headers are sent to the server
        - Authorization headers are transmitted correctly
        - Custom X-* headers are included in the request
        """
        # Arrange - Configure mock server to expect specific headers
        expected_response = {"authenticated": True, "user": "test_user"}

        def handler_with_header_check(request: object) -> Response:
            """Handler that verifies expected headers are present."""
            # werkzeug Request object provides headers access
            auth = request.headers.get("Authorization")  # type: ignore[union-attr]
            request_id = request.headers.get("X-Request-ID")  # type: ignore[union-attr]

            if auth == "Bearer test-token-123" and request_id == "req-456":
                return Response(
                    json.dumps(expected_response),
                    status=200,
                    content_type="application/json",
                )
            return Response(
                json.dumps({"error": "Unauthorized"}),
                status=401,
                content_type="application/json",
            )

        httpserver.expect_request("/api/protected").respond_with_handler(
            handler_with_header_check
        )

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute GET request with custom headers
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {
                    "url": httpserver.url_for("/api/protected"),
                    "headers": {
                        "Authorization": "Bearer test-token-123",
                        "X-Request-ID": "req-456",
                    },
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify successful authenticated response
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 200
            assert result["payload"]["body"] == expected_response

        finally:
            await handler.shutdown()


class TestHttpPostJson:
    """Integration tests for HTTP POST operations with JSON body."""

    async def test_http_post_json(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP POST request with JSON body.

        Verifies that:
        - POST request body is sent correctly
        - Content-Type is set to application/json
        - Server receives and processes the JSON payload
        - Response is properly returned
        """

        # Arrange - Configure mock server to receive POST data
        def echo_handler(request: object) -> Response:
            """Handler that echoes the request body with additional metadata."""
            content_type = request.headers.get("Content-Type", "")  # type: ignore[union-attr]
            body = request.data  # type: ignore[union-attr]

            try:
                request_data = json.loads(body)
            except json.JSONDecodeError:
                request_data = body.decode("utf-8")

            response_data = {
                "received": request_data,
                "content_type": content_type,
                "success": True,
            }
            return Response(
                json.dumps(response_data), status=201, content_type="application/json"
            )

        httpserver.expect_request("/api/users", method="POST").respond_with_handler(
            echo_handler
        )

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute POST request with JSON body
            request_body = {"name": "John Doe", "email": "john@example.com", "age": 30}
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": httpserver.url_for("/api/users"),
                    "body": request_body,
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify response and echoed data
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 201

            response_body = result["payload"]["body"]
            assert response_body["success"] is True
            assert response_body["received"] == request_body
            assert "application/json" in response_body["content_type"]

        finally:
            await handler.shutdown()


class TestHttpPostWithHeaders:
    """Integration tests for HTTP POST operations with custom headers."""

    async def test_http_post_with_headers(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP POST request with custom headers.

        Verifies that:
        - Custom headers are sent with POST request
        - Both Content-Type and custom headers are included
        - Server receives all headers correctly
        """

        # Arrange - Configure mock server to verify headers
        def header_check_handler(request: object) -> Response:
            """Handler that verifies custom headers are present in POST."""
            api_key = request.headers.get("X-API-Key")  # type: ignore[union-attr]
            custom_header = request.headers.get("X-Custom-Header")  # type: ignore[union-attr]

            headers_received = {
                "api_key_present": api_key == "secret-api-key-789",
                "custom_header_present": custom_header == "custom-value",
            }

            return Response(
                json.dumps(headers_received),
                status=200,
                content_type="application/json",
            )

        httpserver.expect_request("/api/submit", method="POST").respond_with_handler(
            header_check_handler
        )

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute POST request with custom headers
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {
                    "url": httpserver.url_for("/api/submit"),
                    "headers": {
                        "X-API-Key": "secret-api-key-789",
                        "X-Custom-Header": "custom-value",
                    },
                    "body": {"data": "test payload"},
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify headers were received correctly
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 200

            response_body = result["payload"]["body"]
            assert response_body["api_key_present"] is True
            assert response_body["custom_header_present"] is True

        finally:
            await handler.shutdown()


class TestHttpGet404:
    """Integration tests for HTTP 404 error responses."""

    async def test_http_get_404(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP GET request handles 404 Not Found response.

        Verifies that:
        - 404 response is returned as successful execution (not exception)
        - Status code is correctly captured as 404
        - Error response body is included in result
        """
        # Arrange - Configure mock server to return 404
        error_response = {"error": "Resource not found", "code": "NOT_FOUND"}
        httpserver.expect_request("/api/nonexistent").respond_with_json(
            error_response, status=404
        )

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute GET request for non-existent resource
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/nonexistent")},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - 404 is returned as successful execution with status code
            # HTTP errors (4xx, 5xx) are valid HTTP responses, not exceptions
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 404
            assert result["payload"]["body"] == error_response

        finally:
            await handler.shutdown()


class TestHttpGet500:
    """Integration tests for HTTP 500 error responses."""

    async def test_http_get_500(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP GET request handles 500 Internal Server Error response.

        Verifies that:
        - 500 response is returned as successful execution (not exception)
        - Status code is correctly captured as 500
        - Server error body is included in result
        """
        # Arrange - Configure mock server to return 500
        error_response = {
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "details": "Database connection failed",
        }
        httpserver.expect_request("/api/broken").respond_with_json(
            error_response, status=500
        )

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute GET request to broken endpoint
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/broken")},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - 500 is returned as successful execution with status code
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 500
            assert result["payload"]["body"] == error_response
            assert result["payload"]["body"]["code"] == "INTERNAL_ERROR"

        finally:
            await handler.shutdown()


class TestHttpTimeout:
    """Integration tests for HTTP timeout handling.

    Note: pytest-httpserver does not support response delays directly.
    These tests use the handler's timeout configuration to test timeout
    behavior with a very short timeout.
    """

    @pytest.mark.skip(
        reason="pytest-httpserver does not support response delays for timeout testing"
    )
    async def test_http_timeout(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP GET request timeout handling.

        Note: This test is skipped because pytest-httpserver does not support
        configuring response delays. For timeout testing, consider using
        aiohttp.test_utils or mocking with controlled delays.

        When enabled, this test verifies that:
        - Timeout raises InfraTimeoutError
        - Error message includes timeout duration
        """
        # This test demonstrates the expected behavior when delays are supported
        # httpserver would need: .respond_with_json(response, delay=35)

        # Initialize handler with very short timeout
        short_timeout_config: dict[str, object] = {
            "max_request_size": 1024 * 1024,
            "max_response_size": 10 * 1024 * 1024,
            # Note: timeout cannot be configured via initialize() in MVP
        }

        handler = HandlerHttpRest()
        await handler.initialize(short_timeout_config)

        try:
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/slow")},
            }

            # Would raise InfraTimeoutError if server had delay > timeout
            with pytest.raises(InfraTimeoutError) as exc_info:
                await handler.execute(envelope)

            assert "timed out" in str(exc_info.value)

        finally:
            await handler.shutdown()


class TestHttpResponseSizeLimit:
    """Integration tests for response size limit enforcement."""

    async def test_http_response_size_limit(
        self, httpserver: HTTPServer, small_response_config: dict[str, object]
    ) -> None:
        """Test HTTP GET enforces response size limits.

        Verifies that:
        - Response larger than max_response_size raises InfraUnavailableError
        - Error message indicates size limit exceeded
        - Size validation uses streaming or Content-Length header
        """
        # Arrange - Configure mock server to return large response
        # 200 bytes exceeds the 100-byte limit in small_response_config
        large_response_data = {"data": "x" * 180}  # ~190 bytes with JSON overhead
        httpserver.expect_request("/api/large").respond_with_json(large_response_data)

        # Initialize handler with small response size limit
        handler = HandlerHttpRest()
        await handler.initialize(small_response_config)

        try:
            # Act - Execute GET request expecting large response
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/large")},
            }

            # Assert - Should raise InfraUnavailableError for size limit
            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            error_msg = str(exc_info.value)
            assert "exceeds configured limit" in error_msg

        finally:
            await handler.shutdown()

    async def test_http_response_within_size_limit_succeeds(
        self, httpserver: HTTPServer, small_response_config: dict[str, object]
    ) -> None:
        """Test HTTP GET succeeds when response is within size limit.

        Verifies that:
        - Response smaller than max_response_size succeeds
        - Full response body is returned correctly
        """
        # Arrange - Configure mock server to return small response
        # Keep response well under 100 bytes
        small_response = {"ok": True}  # ~12 bytes
        httpserver.expect_request("/api/small").respond_with_json(small_response)

        # Initialize handler with small response size limit
        handler = HandlerHttpRest()
        await handler.initialize(small_response_config)

        try:
            # Act - Execute GET request expecting small response
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/small")},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Response should succeed
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 200
            assert result["payload"]["body"] == small_response

        finally:
            await handler.shutdown()


class TestHttpTextResponse:
    """Integration tests for non-JSON HTTP responses."""

    async def test_http_get_text_response(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP GET handles text/plain response correctly.

        Verifies that:
        - Non-JSON content types are returned as text
        - Content-Type header is captured correctly
        """
        # Arrange - Configure mock server to return plain text
        httpserver.expect_request("/api/status").respond_with_data(
            "Service OK", content_type="text/plain"
        )

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute GET request
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/status")},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify text response handling
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 200
            assert result["payload"]["body"] == "Service OK"
            assert "text/plain" in result["payload"]["headers"]["content-type"]

        finally:
            await handler.shutdown()


class TestHttpQueryParameters:
    """Integration tests for HTTP GET with query parameters."""

    async def test_http_get_with_query_params(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP GET with URL query parameters.

        Verifies that:
        - Query parameters in URL are passed correctly
        - Server receives and can process query parameters
        """

        # Arrange - Configure mock server to handle query params
        def query_handler(request: object) -> Response:
            """Handler that returns query parameters."""
            page = request.args.get("page", "1")  # type: ignore[union-attr]
            limit = request.args.get("limit", "10")  # type: ignore[union-attr]
            filter_val = request.args.get("filter", "")  # type: ignore[union-attr]

            response_data = {
                "page": int(page),
                "limit": int(limit),
                "filter": filter_val,
                "total": 100,
            }
            return Response(
                json.dumps(response_data), status=200, content_type="application/json"
            )

        httpserver.expect_request("/api/items").respond_with_handler(query_handler)

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute GET request with query parameters
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {
                    "url": httpserver.url_for("/api/items")
                    + "?page=2&limit=25&filter=active"
                },
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify query params were processed
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 200

            body = result["payload"]["body"]
            assert body["page"] == 2
            assert body["limit"] == 25
            assert body["filter"] == "active"

        finally:
            await handler.shutdown()


class TestHttpEmptyResponse:
    """Integration tests for HTTP responses with empty body."""

    async def test_http_get_empty_body(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP GET handles empty response body (204 No Content).

        Verifies that:
        - Empty response body is handled correctly
        - Status code 204 is captured
        """
        # Arrange - Configure mock server to return 204 with no body
        httpserver.expect_request("/api/ping").respond_with_data("", status=204)

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute GET request
            envelope: dict[str, object] = {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/api/ping")},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify empty response handling
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 204
            assert result["payload"]["body"] == ""

        finally:
            await handler.shutdown()


class TestHttpPostEmptyBody:
    """Integration tests for HTTP POST with no body."""

    async def test_http_post_no_body(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test HTTP POST request with no body (trigger/webhook style).

        Verifies that:
        - POST without body is accepted
        - Server receives the request correctly
        """
        # Arrange - Configure mock server
        httpserver.expect_request("/api/trigger", method="POST").respond_with_json(
            {"triggered": True, "timestamp": "2025-01-01T00:00:00Z"}, status=200
        )

        # Initialize handler
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute POST request without body
            envelope: dict[str, object] = {
                "operation": "http.post",
                "payload": {"url": httpserver.url_for("/api/trigger")},
            }

            output = await handler.execute(envelope)
            result = output.result

            # Assert - Verify response
            assert result["status"] == "success"
            assert result["payload"]["status_code"] == 200
            assert result["payload"]["body"]["triggered"] is True

        finally:
            await handler.shutdown()


class TestHttpMultipleRequests:
    """Integration tests for multiple sequential HTTP requests."""

    async def test_http_multiple_requests_reuse_handler(
        self, httpserver: HTTPServer, http_handler_config: dict[str, object]
    ) -> None:
        """Test multiple HTTP requests using same handler instance.

        Verifies that:
        - Handler can process multiple requests
        - Each request gets correct response
        - Handler state is properly maintained
        """
        # Arrange - Configure multiple endpoints
        httpserver.expect_request("/api/first").respond_with_json({"order": 1})
        httpserver.expect_request("/api/second").respond_with_json({"order": 2})
        httpserver.expect_request("/api/third").respond_with_json({"order": 3})

        # Initialize handler once
        handler = HandlerHttpRest()
        await handler.initialize(http_handler_config)

        try:
            # Act - Execute multiple requests
            results = []
            for path in ["/api/first", "/api/second", "/api/third"]:
                envelope: dict[str, object] = {
                    "operation": "http.get",
                    "payload": {"url": httpserver.url_for(path)},
                }
                output = await handler.execute(envelope)
                results.append(output.result)

            # Assert - All requests succeeded in order
            assert len(results) == 3
            assert all(r["status"] == "success" for r in results)
            assert results[0]["payload"]["body"]["order"] == 1
            assert results[1]["payload"]["body"]["order"] == 2
            assert results[2]["payload"]["body"]["order"] == 3

        finally:
            await handler.shutdown()


__all__: list[str] = [
    "TestHttpGetSuccess",
    "TestHttpGetWithHeaders",
    "TestHttpPostJson",
    "TestHttpPostWithHeaders",
    "TestHttpGet404",
    "TestHttpGet500",
    "TestHttpTimeout",
    "TestHttpResponseSizeLimit",
    "TestHttpTextResponse",
    "TestHttpQueryParameters",
    "TestHttpEmptyResponse",
    "TestHttpPostEmptyBody",
    "TestHttpMultipleRequests",
]

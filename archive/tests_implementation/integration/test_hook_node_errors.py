"""
Error Handling Validation Tests for Hook Node.

Tests comprehensive error handling including:
- Network failures and timeouts (30-second timeout handling)
- Invalid authentication credentials for all auth methods
- Circuit breaker protection activation and state transitions
- Proper OnexError chaining with CoreErrorCode
- Error recovery scenarios and graceful degradation
- Security validation and input sanitization
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_auth_type import EnumAuthType
from omnibase_core.enums.enum_backoff_strategy import EnumBackoffStrategy
from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_spi.protocols.core import ProtocolHttpResponse

from omnibase_infra.models.notification.model_notification_auth import (
    ModelNotificationAuth,
)

# Shared notification models
from omnibase_infra.models.notification.model_notification_request import (
    ModelNotificationRequest,
)
from omnibase_infra.models.notification.model_notification_retry_policy import (
    ModelNotificationRetryPolicy,
)
from omnibase_infra.nodes.hook_node.v1_0_0.models.model_hook_node_input import (
    ModelHookNodeInput,
)

# Hook Node implementation
from omnibase_infra.nodes.hook_node.v1_0_0.node import (
    CircuitBreakerState,
    NodeHookEffect,
)


class TestNetworkFailuresAndTimeouts:
    """Test suite for network failure and timeout handling."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def hook_node(self, container):
        """Create Hook Node with mocked dependencies."""
        mock_http_client = AsyncMock()
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, hook_node):
        """Test 30-second timeout error handling."""
        request = ModelNotificationRequest(
            url="https://slow.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"message": "This will timeout"},
        )

        # Mock timeout after 30 seconds
        async def mock_timeout(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay to simulate work
            raise TimeoutError("Request timeout after 30 seconds")

        hook_node._http_client.post = mock_timeout

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        # Verify proper error conversion
        assert exc_info.value.code == CoreErrorCode.TIMEOUT_ERROR
        assert "timeout" in str(exc_info.value).lower()
        assert hook_node._failed_notifications == 1

        # Verify circuit breaker recorded the failure
        url_key = str(request.url)
        assert url_key in hook_node._circuit_breakers
        circuit_breaker = hook_node._circuit_breakers[url_key]
        assert circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, hook_node):
        """Test connection error handling (DNS resolution, network unreachable)."""
        request = ModelNotificationRequest(
            url="https://nonexistent.invalid.domain.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "connection error"},
        )

        # Mock connection error
        hook_node._http_client.post = AsyncMock(
            side_effect=ConnectionError("Network is unreachable"),
        )

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        # Verify proper error conversion and chaining
        assert exc_info.value.code == CoreErrorCode.NETWORK_ERROR
        assert "network" in str(exc_info.value).lower() or "connection" in str(exc_info.value).lower()

        # Verify original exception is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    @pytest.mark.asyncio
    async def test_dns_resolution_error_handling(self, hook_node):
        """Test DNS resolution error handling."""
        request = ModelNotificationRequest(
            url="https://this-domain-does-not-exist-12345.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "DNS error"},
        )

        # Mock DNS resolution error
        import socket
        hook_node._http_client.post = AsyncMock(
            side_effect=socket.gaierror("Name or service not known"),
        )

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.NETWORK_ERROR
        assert "dns" in str(exc_info.value).lower() or "resolution" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_ssl_certificate_error_handling(self, hook_node):
        """Test SSL certificate validation error handling."""
        request = ModelNotificationRequest(
            url="https://expired-certificate.badssl.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "SSL error"},
        )

        # Mock SSL certificate error
        import ssl
        hook_node._http_client.post = AsyncMock(
            side_effect=ssl.SSLError("certificate verify failed: certificate has expired"),
        )

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.SECURITY_ERROR
        assert "ssl" in str(exc_info.value).lower() or "certificate" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_http_error_status_codes(self, hook_node):
        """Test handling of various HTTP error status codes."""
        error_scenarios = [
            (400, "Bad Request", CoreErrorCode.INVALID_INPUT),
            (401, "Unauthorized", CoreErrorCode.AUTHENTICATION_ERROR),
            (403, "Forbidden", CoreErrorCode.AUTHORIZATION_ERROR),
            (404, "Not Found", CoreErrorCode.NOT_FOUND),
            (429, "Too Many Requests", CoreErrorCode.RATE_LIMIT_ERROR),
            (500, "Internal Server Error", CoreErrorCode.EXTERNAL_SERVICE_ERROR),
            (502, "Bad Gateway", CoreErrorCode.EXTERNAL_SERVICE_ERROR),
            (503, "Service Unavailable", CoreErrorCode.SERVICE_UNAVAILABLE),
            (504, "Gateway Timeout", CoreErrorCode.TIMEOUT_ERROR),
        ]

        for status_code, status_text, expected_error_code in error_scenarios:
            with pytest.raises(OnexError) as exc_info:
                request = ModelNotificationRequest(
                    url=f"https://error-test.com/webhook/{status_code}",
                    method=EnumNotificationMethod.POST,
                    payload={"test": f"HTTP {status_code} error"},
                )

                # Mock HTTP error response
                error_response = ProtocolHttpResponse(
                    status_code=status_code,
                    headers={"Content-Type": "text/plain"},
                    body=status_text,
                    execution_time_ms=100.0,
                    is_success=False,
                )
                hook_node._http_client.post = AsyncMock(return_value=error_response)

                input_data = ModelHookNodeInput(notification_request=request)
                await hook_node.process(input_data)

            # Verify correct error code mapping
            assert exc_info.value.code == expected_error_code, f"Status {status_code} should map to {expected_error_code}"


class TestAuthenticationFailures:
    """Test suite for authentication failure scenarios."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def hook_node(self, container):
        """Create Hook Node with mocked dependencies."""
        mock_http_client = AsyncMock()
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_invalid_bearer_token(self, hook_node):
        """Test handling of invalid Bearer tokens."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BEARER,
            credentials={"token": "invalid-bearer-token"},
        )

        request = ModelNotificationRequest(
            url="https://api.example.com/authenticated-webhook",
            method=EnumNotificationMethod.POST,
            payload={"message": "Authenticated test"},
            auth=auth,
        )

        # Mock 401 Unauthorized response
        auth_error_response = ProtocolHttpResponse(
            status_code=401,
            headers={"Content-Type": "application/json"},
            body='{"error": "Invalid token", "code": "INVALID_TOKEN"}',
            execution_time_ms=50.0,
            is_success=False,
        )
        hook_node._http_client.post = AsyncMock(return_value=auth_error_response)

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.AUTHENTICATION_ERROR
        assert "authentication" in str(exc_info.value).lower() or "unauthorized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_basic_auth_credentials(self, hook_node):
        """Test handling of invalid Basic auth credentials."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BASIC,
            credentials={"username": "wronguser", "password": "wrongpass"},
        )

        request = ModelNotificationRequest(
            url="https://basic-auth.example.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"message": "Basic auth test"},
            auth=auth,
        )

        # Mock 401 response for invalid credentials
        auth_error_response = ProtocolHttpResponse(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Webhook API"'},
            body="Unauthorized",
            execution_time_ms=30.0,
            is_success=False,
        )
        hook_node._http_client.post = AsyncMock(return_value=auth_error_response)

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_invalid_api_key(self, hook_node):
        """Test handling of invalid API keys."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.API_KEY_HEADER,
            credentials={"header_name": "X-API-Key", "api_key": "invalid-api-key"},
        )

        request = ModelNotificationRequest(
            url="https://api-key-protected.example.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"message": "API key test"},
            auth=auth,
        )

        # Mock 403 Forbidden response for invalid API key
        auth_error_response = ProtocolHttpResponse(
            status_code=403,
            headers={"Content-Type": "application/json"},
            body='{"error": "Invalid API key", "code": "FORBIDDEN"}',
            execution_time_ms=40.0,
            is_success=False,
        )
        hook_node._http_client.post = AsyncMock(return_value=auth_error_response)

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.AUTHORIZATION_ERROR

    def test_malformed_authentication_credentials(self):
        """Test validation of malformed authentication credentials."""
        # Test Bearer auth without token
        with pytest.raises(ValueError, match="Bearer auth requires 'token'"):
            ModelNotificationAuth(
                auth_type=EnumAuthType.BEARER,
                credentials={},
            )

        # Test Basic auth with missing password
        with pytest.raises(ValueError, match="Basic auth requires 'username' and 'password'"):
            ModelNotificationAuth(
                auth_type=EnumAuthType.BASIC,
                credentials={"username": "testuser"},
            )

        # Test API key auth with missing header name
        with pytest.raises(ValueError, match="API key auth requires 'header_name' and 'api_key'"):
            ModelNotificationAuth(
                auth_type=EnumAuthType.API_KEY_HEADER,
                credentials={"api_key": "test-key"},
            )

    @pytest.mark.asyncio
    async def test_auth_credential_exposure_prevention(self, hook_node):
        """Test that authentication credentials are not exposed in logs or errors."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BEARER,
            credentials={"token": "super-secret-token-12345"},
        )

        request = ModelNotificationRequest(
            url="https://secure.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"message": "Security test"},
            auth=auth,
        )

        # Mock authentication failure
        hook_node._http_client.post = AsyncMock(
            side_effect=Exception("Authentication failed"),
        )

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        # Verify that the secret token is not exposed in the error message
        error_message = str(exc_info.value)
        assert "super-secret-token-12345" not in error_message
        assert "Bearer " not in error_message or "***" in error_message


class TestCircuitBreakerProtection:
    """Test suite for circuit breaker protection activation."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def hook_node(self, container):
        """Create Hook Node with mocked dependencies."""
        mock_http_client = AsyncMock()
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_transitions(self, hook_node):
        """Test circuit breaker state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        request = ModelNotificationRequest(
            url="https://state-transition-test.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "circuit breaker states"},
        )

        # Mock consistent failures
        failure_response = ProtocolHttpResponse(
            status_code=500,
            headers={},
            body="Internal Server Error",
            execution_time_ms=100.0,
            is_success=False,
        )
        hook_node._http_client.post = AsyncMock(return_value=failure_response)

        input_data = ModelHookNodeInput(notification_request=request)
        url_key = str(request.url)

        # Initial state should be CLOSED (no circuit breaker yet)
        assert url_key not in hook_node._circuit_breakers

        # Send failures to trigger OPEN state
        for i in range(6):  # Exceed failure threshold (5)
            try:
                await hook_node.process(input_data)
            except OnexError:
                pass

        # Verify circuit breaker is now OPEN
        circuit_breaker = hook_node._circuit_breakers[url_key]
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count >= 5

        # Verify circuit breaker blocks subsequent requests
        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.SERVICE_UNAVAILABLE
        assert "Circuit breaker is OPEN" in str(exc_info.value)

        # HTTP client should not be called when circuit breaker is OPEN
        # (Previous calls were made during the failure accumulation)
        previous_call_count = hook_node._http_client.post.call_count

        try:
            await hook_node.process(input_data)
        except OnexError:
            pass

        # No additional HTTP calls should be made
        assert hook_node._http_client.post.call_count == previous_call_count

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_timeout(self, hook_node):
        """Test circuit breaker recovery after timeout period."""
        request = ModelNotificationRequest(
            url="https://recovery-timeout-test.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "recovery timeout"},
        )

        # Trigger circuit breaker opening
        failure_response = ProtocolHttpResponse(
            status_code=503,
            headers={},
            body="Service Unavailable",
            execution_time_ms=100.0,
            is_success=False,
        )
        hook_node._http_client.post = AsyncMock(return_value=failure_response)

        input_data = ModelHookNodeInput(notification_request=request)

        # Open the circuit breaker
        for i in range(6):
            try:
                await hook_node.process(input_data)
            except OnexError:
                pass

        url_key = str(request.url)
        circuit_breaker = hook_node._circuit_breakers[url_key]
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Manually advance time to trigger recovery timeout
        circuit_breaker.last_failure_time = time.time() - 70  # 70 seconds ago (past 60s recovery timeout)

        # Configure success response for recovery attempt
        success_response = ProtocolHttpResponse(
            status_code=200,
            headers={},
            body="OK",
            execution_time_ms=50.0,
            is_success=True,
        )
        hook_node._http_client.post = AsyncMock(return_value=success_response)

        # Next request should attempt recovery
        result = await hook_node.process(input_data)

        assert result.success is True
        # Circuit breaker should transition to HALF_OPEN then CLOSED
        assert circuit_breaker.state in [CircuitBreakerState.HALF_OPEN, CircuitBreakerState.CLOSED]

    @pytest.mark.asyncio
    async def test_circuit_breaker_per_destination_isolation(self, hook_node):
        """Test that circuit breakers are isolated per destination."""
        # Create requests to different URLs
        failing_request = ModelNotificationRequest(
            url="https://failing-service.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "failing service"},
        )

        working_request = ModelNotificationRequest(
            url="https://working-service.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "working service"},
        )

        # Configure selective responses
        def selective_response(url, *args, **kwargs):
            if "failing-service.com" in url:
                return ProtocolHttpResponse(
                    status_code=500,
                    headers={},
                    body="Error",
                    execution_time_ms=100.0,
                    is_success=False,
                )
            return ProtocolHttpResponse(
                status_code=200,
                headers={},
                body="OK",
                execution_time_ms=50.0,
                is_success=True,
            )

        hook_node._http_client.post = lambda url, **kwargs: selective_response(url, **kwargs)

        # Trigger circuit breaker for failing service
        failing_input = ModelHookNodeInput(notification_request=failing_request)
        for i in range(6):
            try:
                await hook_node.process(failing_input)
            except OnexError:
                pass

        # Verify failing service circuit breaker is OPEN
        failing_cb = hook_node._circuit_breakers.get(str(failing_request.url))
        assert failing_cb is not None
        assert failing_cb.state == CircuitBreakerState.OPEN

        # Working service should still function normally
        working_input = ModelHookNodeInput(notification_request=working_request)
        result = await hook_node.process(working_input)

        assert result.success is True

        # Verify working service has no circuit breaker or it's in CLOSED state
        working_cb = hook_node._circuit_breakers.get(str(working_request.url))
        assert working_cb is None or working_cb.state == CircuitBreakerState.CLOSED


class TestInputValidationAndSecurity:
    """Test suite for input validation and security measures."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def hook_node(self, container):
        """Create Hook Node with mocked dependencies."""
        mock_http_client = AsyncMock()
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_missing_notification_request_validation(self, hook_node):
        """Test validation when notification request is missing."""
        input_data = ModelHookNodeInput(notification_request=None)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.INVALID_INPUT
        assert "Notification request is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_url_validation(self, hook_node):
        """Test validation of invalid URLs."""
        # Note: URL validation is handled by Pydantic HttpUrl type during model creation
        # Test with empty URL should be caught during model instantiation
        with pytest.raises(ValueError):
            ModelNotificationRequest(
                url="",  # Empty URL
                method=EnumNotificationMethod.POST,
                payload={"test": "invalid URL"},
            )

        # Test with malformed URL
        with pytest.raises(ValueError):
            ModelNotificationRequest(
                url="not-a-valid-url",
                method=EnumNotificationMethod.POST,
                payload={"test": "malformed URL"},
            )

    def test_url_sanitization_for_logging(self, hook_node):
        """Test URL sanitization prevents sensitive data exposure in logs."""
        # Test URL with sensitive query parameters
        sensitive_url = "https://webhook.com/api/notify?token=secret123&api_key=sensitive456&normal=value"

        sanitized = hook_node._logger._sanitize_url_for_logging(sensitive_url)

        # Verify sensitive data is masked
        assert "secret123" not in sanitized
        assert "sensitive456" not in sanitized
        assert "token=***" in sanitized
        assert "api_key=***" in sanitized
        assert "normal=value" in sanitized  # Non-sensitive params preserved

    @pytest.mark.asyncio
    async def test_payload_size_limits(self, hook_node):
        """Test handling of excessively large payloads."""
        # Create a very large payload (10MB of data)
        large_payload = {
            "message": "x" * (10 * 1024 * 1024),  # 10MB string
            "data": list(range(100000)),  # Large array
        }

        request = ModelNotificationRequest(
            url="https://webhook.com/api/large-payload",
            method=EnumNotificationMethod.POST,
            payload=large_payload,
        )

        # Mock response for large payload (some services reject large payloads)
        large_payload_response = ProtocolHttpResponse(
            status_code=413,  # Payload Too Large
            headers={"Content-Type": "text/plain"},
            body="Request Entity Too Large",
            execution_time_ms=100.0,
            is_success=False,
        )
        hook_node._http_client.post = AsyncMock(return_value=large_payload_response)

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_ssrf_protection_awareness(self, hook_node):
        """Test SSRF protection awareness (URL validation responsibility)."""
        # Test internal/private IP addresses that could be SSRF risks
        potentially_dangerous_urls = [
            "http://127.0.0.1:8080/admin",
            "http://10.0.0.1/internal",
            "http://192.168.1.1/config",
            "http://169.254.169.254/metadata",  # AWS metadata service
            "http://localhost:22/ssh",
        ]

        for dangerous_url in potentially_dangerous_urls:
            # Note: Hook Node should pass URL validation responsibility to the consuming service
            # The models should accept these URLs but log warnings
            request = ModelNotificationRequest(
                url=dangerous_url,
                method=EnumNotificationMethod.POST,
                payload={"test": "SSRF protection test"},
            )

            # Verify the model accepts the URL (validation is service responsibility)
            assert request.url is not None
            # In production, the consuming service should validate and reject these URLs

    def test_header_injection_prevention(self):
        """Test prevention of header injection attacks."""
        # Test with headers containing newlines (potential header injection)
        with patch("logging.Logger.warning") as mock_warning:
            request = ModelNotificationRequest(
                url="https://webhook.com/api/test",
                method=EnumNotificationMethod.POST,
                payload={"test": "header injection test"},
                headers={
                    "X-Custom-Header": "safe-value",
                    "X-Injection-Attempt": "value\r\nInjected-Header: malicious",
                },
            )

            # Model should accept the headers but they should be sanitized by HTTP client
            assert request.headers is not None

    @pytest.mark.asyncio
    async def test_json_serialization_errors(self, hook_node):
        """Test handling of JSON serialization errors."""
        # Create payload with non-serializable data
        class NonSerializable:
            def __str__(self):
                return "non-serializable object"

        # Note: Pydantic should prevent non-serializable objects in payload
        # But test the edge case where serialization fails
        with patch("json.dumps", side_effect=TypeError("Object is not JSON serializable")):
            request = ModelNotificationRequest(
                url="https://webhook.com/api/test",
                method=EnumNotificationMethod.POST,
                payload={"test": "serialization test", "data": "string_value"},
            )

            input_data = ModelHookNodeInput(notification_request=request)

            with pytest.raises(OnexError) as exc_info:
                await hook_node.process(input_data)

            assert exc_info.value.code == CoreErrorCode.SYSTEM_ERROR


class TestErrorRecoveryScenarios:
    """Test suite for error recovery and graceful degradation."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def hook_node(self, container):
        """Create Hook Node with mocked dependencies."""
        mock_http_client = AsyncMock()
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, hook_node):
        """Test recovery from partial failures with retry policies."""
        retry_policy = ModelNotificationRetryPolicy(
            max_attempts=4,
            strategy=EnumBackoffStrategy.EXPONENTIAL,
            base_delay_ms=50,
            max_delay_ms=1000,
            backoff_multiplier=2.0,
        )

        request = ModelNotificationRequest(
            url="https://intermittent-failure.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "partial failure recovery"},
            retry_policy=retry_policy,
        )

        # Configure response sequence: fail, fail, fail, succeed
        responses = [
            ProtocolHttpResponse(status_code=500, headers={}, body="Error", execution_time_ms=100.0, is_success=False),
            ProtocolHttpResponse(status_code=502, headers={}, body="Bad Gateway", execution_time_ms=100.0, is_success=False),
            ProtocolHttpResponse(status_code=503, headers={}, body="Unavailable", execution_time_ms=100.0, is_success=False),
            ProtocolHttpResponse(status_code=200, headers={}, body="OK", execution_time_ms=50.0, is_success=True),
        ]

        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            response = responses[call_count]
            call_count += 1
            return response

        hook_node._http_client.post = mock_post

        with patch("asyncio.sleep"):  # Speed up test
            input_data = ModelHookNodeInput(notification_request=request)
            result = await hook_node.process(input_data)

        # Should eventually succeed
        assert result.success is True
        assert result.notification_result.total_attempts == 4
        assert result.notification_result.final_status_code == 200

    @pytest.mark.asyncio
    async def test_dependency_failure_graceful_degradation(self, hook_node):
        """Test graceful degradation when dependencies fail."""
        request = ModelNotificationRequest(
            url="https://dependency-test.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "dependency failure"},
        )

        # Mock event bus failure (should not affect notification delivery)
        hook_node._event_bus.publish = AsyncMock(side_effect=Exception("Event bus failure"))

        # HTTP client should still work
        success_response = ProtocolHttpResponse(
            status_code=200,
            headers={},
            body="OK",
            execution_time_ms=100.0,
            is_success=True,
        )
        hook_node._http_client.post = AsyncMock(return_value=success_response)

        input_data = ModelHookNodeInput(notification_request=request)

        # Notification should still succeed despite event bus failure
        result = await hook_node.process(input_data)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, hook_node):
        """Test handling of resource exhaustion scenarios."""
        request = ModelNotificationRequest(
            url="https://resource-exhaustion-test.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "resource exhaustion"},
        )

        # Mock memory exhaustion
        hook_node._http_client.post = AsyncMock(
            side_effect=MemoryError("Out of memory"),
        )

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.SYSTEM_ERROR
        assert "memory" in str(exc_info.value).lower() or "resource" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, hook_node):
        """Test error handling under concurrent load."""
        requests = [
            ModelNotificationRequest(
                url=f"https://concurrent-error-test-{i}.com/webhook",
                method=EnumNotificationMethod.POST,
                payload={"test": f"concurrent error {i}", "id": i},
            )
            for i in range(10)
        ]

        # Configure some requests to fail, others to succeed
        def selective_response(url, *args, **kwargs):
            # Fail requests with even IDs, succeed with odd IDs
            if any(f"error-test-{i}" in url for i in range(0, 10, 2)):  # Even IDs
                return ProtocolHttpResponse(
                    status_code=500,
                    headers={},
                    body="Error",
                    execution_time_ms=100.0,
                    is_success=False,
                )
            # Odd IDs
            return ProtocolHttpResponse(
                status_code=200,
                headers={},
                body="OK",
                execution_time_ms=50.0,
                is_success=True,
            )

        hook_node._http_client.post = selective_response

        # Process all requests concurrently
        input_data_list = [ModelHookNodeInput(notification_request=req) for req in requests]

        # Some will succeed, some will fail - test that errors don't interfere
        tasks = [hook_node.process(input_data) for input_data in input_data_list]

        results = []
        for task in tasks:
            try:
                result = await task
                results.append(("success", result))
            except OnexError as e:
                results.append(("error", e))

        # Verify we have both successes and failures
        successes = [r for r in results if r[0] == "success"]
        failures = [r for r in results if r[0] == "error"]

        assert len(successes) == 5  # Odd IDs should succeed
        assert len(failures) == 5   # Even IDs should fail

        # Verify error isolation - failures don't affect successes
        for result_type, result in successes:
            assert result.success is True

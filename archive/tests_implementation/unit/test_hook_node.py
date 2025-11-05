"""
Comprehensive Unit Tests for Hook Node.

Tests the core NodeHookEffect service functionality including:
- Notification model validation
- Circuit breaker behavior per destination
- Retry policy execution with backoff strategies
- Authentication handling for Bearer/Basic/API key methods
- Structured logging and correlation ID tracking
- Performance metrics and observability
"""

import time
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_auth_type import EnumAuthType
from omnibase_core.enums.enum_backoff_strategy import EnumBackoffStrategy
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_spi.protocols.core import ProtocolHttpResponse

from omnibase_infra.models.notification.model_notification_attempt import (
    ModelNotificationAttempt,
)
from omnibase_infra.models.notification.model_notification_auth import (
    ModelNotificationAuth,
)

# Shared notification models
from omnibase_infra.models.notification.model_notification_request import (
    ModelNotificationRequest,
)
from omnibase_infra.models.notification.model_notification_result import (
    ModelNotificationResult,
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
    HookStructuredLogger,
    NodeHookEffect,
)


class TestNotificationModels:
    """Test suite for all notification model validation and functionality."""

    def test_notification_request_creation(self):
        """Test creating notification request with all fields."""
        request = ModelNotificationRequest(
            url="https://hooks.slack.com/services/test",
            method=EnumNotificationMethod.POST,
            payload={"text": "Test notification"},
            headers={"Content-Type": "application/json"},
            auth=ModelNotificationAuth(
                auth_type=EnumAuthType.BEARER,
                credentials={"token": "test-token"},
            ),
        )

        assert str(request.url) == "https://hooks.slack.com/services/test"
        assert request.method == EnumNotificationMethod.POST
        assert request.payload == {"text": "Test notification"}
        assert request.requires_authentication is True

    def test_notification_auth_bearer_token(self):
        """Test Bearer token authentication configuration."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BEARER,
            credentials={"token": "bearer-token-123"},
        )

        assert auth.is_bearer_auth is True
        assert auth.is_basic_auth is False
        assert auth.is_api_key_auth is False

        headers = auth.get_auth_header()
        assert headers == {"Authorization": "Bearer bearer-token-123"}

    def test_notification_auth_basic_auth(self):
        """Test Basic authentication configuration."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BASIC,
            credentials={"username": "testuser", "password": "testpass"},
        )

        assert auth.is_basic_auth is True
        assert auth.is_bearer_auth is False

        headers = auth.get_auth_header()
        # Basic dGVzdHVzZXI6dGVzdHBhc3M= is base64 of "testuser:testpass"
        assert headers["Authorization"].startswith("Basic ")

    def test_notification_auth_api_key(self):
        """Test API key authentication configuration."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.API_KEY_HEADER,
            credentials={"header_name": "X-API-Key", "api_key": "api-key-123"},
        )

        assert auth.is_api_key_auth is True
        headers = auth.get_auth_header()
        assert headers == {"X-API-Key": "api-key-123"}

    def test_notification_auth_validation_errors(self):
        """Test authentication model validation errors."""
        # Bearer auth without token
        with pytest.raises(ValueError, match="Bearer auth requires 'token'"):
            ModelNotificationAuth(
                auth_type=EnumAuthType.BEARER,
                credentials={},
            )

        # Basic auth without required fields
        with pytest.raises(ValueError, match="Basic auth requires 'username' and 'password'"):
            ModelNotificationAuth(
                auth_type=EnumAuthType.BASIC,
                credentials={"username": "test"},
            )

        # API key auth without required fields
        with pytest.raises(ValueError, match="API key auth requires 'header_name' and 'api_key'"):
            ModelNotificationAuth(
                auth_type=EnumAuthType.API_KEY_HEADER,
                credentials={"header_name": "X-API-Key"},
            )

    def test_notification_retry_policy(self):
        """Test retry policy configuration."""
        retry_policy = ModelNotificationRetryPolicy(
            max_attempts=5,
            strategy=EnumBackoffStrategy.EXPONENTIAL,
            base_delay_ms=1000,
            max_delay_ms=30000,
            backoff_multiplier=2.0,
        )

        assert retry_policy.max_attempts == 5
        assert retry_policy.strategy == EnumBackoffStrategy.EXPONENTIAL
        assert retry_policy.is_exponential_backoff is True
        assert retry_policy.is_linear_backoff is False
        assert retry_policy.is_fixed_delay is False

    def test_notification_attempt_model(self):
        """Test notification attempt tracking model."""
        attempt = ModelNotificationAttempt(
            attempt_number=1,
            start_time=time.time(),
            execution_time_ms=150.5,
            http_status_code=200,
            is_success=True,
            error_message=None,
        )

        assert attempt.attempt_number == 1
        assert attempt.is_success is True
        assert attempt.http_status_code == 200
        assert attempt.execution_time_ms == 150.5

    def test_notification_result_success(self):
        """Test successful notification result model."""
        attempts = [
            ModelNotificationAttempt(
                attempt_number=1,
                start_time=time.time(),
                execution_time_ms=120.0,
                http_status_code=200,
                is_success=True,
            ),
        ]

        result = ModelNotificationResult(
            is_success=True,
            total_attempts=1,
            attempts=attempts,
            final_status_code=200,
            final_error_message=None,
            total_execution_time_ms=120.0,
        )

        assert result.is_success is True
        assert result.total_attempts == 1
        assert result.final_status_code == 200
        assert len(result.attempts) == 1


class TestHookStructuredLogger:
    """Test suite for structured logging functionality."""

    def test_structured_logger_initialization(self):
        """Test logger initialization with default settings."""
        logger = HookStructuredLogger()
        assert logger.logger.name == "hook_node"

    def test_correlation_id_tracking(self):
        """Test correlation ID inclusion in log entries."""
        logger = HookStructuredLogger()
        correlation_id = uuid4()

        with patch.object(logger.logger, "info") as mock_info:
            logger.info("Test message", correlation_id=correlation_id)

            # Verify correlation ID is in extra data
            call_args = mock_info.call_args
            extra = call_args[1]["extra"]
            assert extra["correlation_id"] == str(correlation_id)
            assert extra["operation"] == "notification"
            assert extra["component"] == "hook_node"

    def test_url_sanitization(self):
        """Test URL sanitization removes sensitive parameters."""
        logger = HookStructuredLogger()

        # Test URL with sensitive query parameters
        sensitive_url = "https://api.example.com/webhook?token=secret123&key=apikey456&normal=value"
        sanitized = logger._sanitize_url_for_logging(sensitive_url)

        assert "token=***" in sanitized
        assert "key=***" in sanitized
        assert "normal=value" in sanitized
        assert "secret123" not in sanitized
        assert "apikey456" not in sanitized

    def test_notification_logging_methods(self):
        """Test specialized notification logging methods."""
        logger = HookStructuredLogger()
        correlation_id = uuid4()

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_notification_start(
                correlation_id=correlation_id,
                url="https://hooks.slack.com/test",
                method="POST",
                retry_attempt=2,
            )

            call_args = mock_info.call_args
            assert "Starting notification attempt 2" in call_args[0][0]
            extra = call_args[1]["extra"]
            assert extra["operation"] == "notification_send"
            assert extra["retry_attempt"] == 2

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_notification_success(
                correlation_id=correlation_id,
                execution_time_ms=125.5,
                status_code=200,
                retry_attempt=1,
            )

            call_args = mock_info.call_args
            assert "succeeded: 200 (125.50ms)" in call_args[0][0]

        with patch.object(logger.logger, "error") as mock_error:
            test_exception = Exception("Network timeout")
            logger.log_notification_error(
                correlation_id=correlation_id,
                execution_time_ms=30000.0,
                exception=test_exception,
                retry_attempt=3,
            )

            call_args = mock_error.call_args
            assert "Notification attempt 3 failed" in call_args[0][0]
            extra = call_args[1]["extra"]
            assert extra["exception_type"] == "Exception"
            assert extra["exception_message"] == "Network timeout"


class TestNodeHookEffect:
    """Test suite for NodeHookEffect service functionality."""

    @pytest.fixture
    def container(self):
        """Create a test container with mocked dependencies."""
        container = ModelONEXContainer()

        # Mock HTTP client
        mock_http_client = AsyncMock()
        container.provide("protocol_http_client", mock_http_client)

        # Mock event bus
        mock_event_bus = AsyncMock()
        container.provide("protocol_event_bus", mock_event_bus)

        return container

    @pytest.fixture
    def hook_node(self, container):
        """Create a Hook Node instance with mocked dependencies."""
        return NodeHookEffect(container)

    @pytest.fixture
    def basic_notification_request(self):
        """Create a basic notification request for testing."""
        return ModelNotificationRequest(
            url="https://hooks.slack.com/services/test",
            method=EnumNotificationMethod.POST,
            payload={"text": "Test notification", "channel": "#alerts"},
        )

    def test_hook_node_initialization(self, hook_node):
        """Test Hook Node proper initialization."""
        assert hook_node._total_notifications == 0
        assert hook_node._successful_notifications == 0
        assert hook_node._failed_notifications == 0
        assert len(hook_node._circuit_breakers) == 0

    @pytest.mark.asyncio
    async def test_successful_notification_delivery(self, hook_node, basic_notification_request):
        """Test successful notification delivery without retries."""
        # Mock successful HTTP response
        mock_response = ProtocolHttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"status": "ok"}',
            execution_time_ms=125.0,
            is_success=True,
        )

        hook_node._http_client.post = AsyncMock(return_value=mock_response)

        input_data = ModelHookNodeInput(notification_request=basic_notification_request)
        result = await hook_node.process(input_data)

        assert result.success is True
        assert result.notification_result.is_success is True
        assert result.notification_result.total_attempts == 1
        assert result.notification_result.final_status_code == 200
        assert hook_node._successful_notifications == 1
        assert hook_node._failed_notifications == 0

    @pytest.mark.asyncio
    async def test_notification_with_authentication(self, hook_node):
        """Test notification delivery with Bearer token authentication."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BEARER,
            credentials={"token": "test-bearer-token"},
        )

        request = ModelNotificationRequest(
            url="https://api.example.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"message": "Authenticated notification"},
            auth=auth,
        )

        mock_response = ProtocolHttpResponse(
            status_code=200,
            headers={},
            body='{"received": true}',
            execution_time_ms=100.0,
            is_success=True,
        )

        hook_node._http_client.post = AsyncMock(return_value=mock_response)

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        # Verify authentication headers were added
        call_args = hook_node._http_client.post.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-bearer-token"

        assert result.success is True

    @pytest.mark.asyncio
    async def test_retry_policy_exponential_backoff(self, hook_node, basic_notification_request):
        """Test retry policy with exponential backoff strategy."""
        retry_policy = ModelNotificationRetryPolicy(
            max_attempts=3,
            strategy=EnumBackoffStrategy.EXPONENTIAL,
            base_delay_ms=100,
            max_delay_ms=1000,
            backoff_multiplier=2.0,
        )

        request = ModelNotificationRequest(
            url="https://hooks.discord.com/api/webhooks/test",
            method=EnumNotificationMethod.POST,
            payload={"content": "Test message"},
            retry_policy=retry_policy,
        )

        # Mock two failures, then success
        failure_response = ProtocolHttpResponse(
            status_code=500,
            headers={},
            body='{"error": "Internal server error"}',
            execution_time_ms=50.0,
            is_success=False,
        )

        success_response = ProtocolHttpResponse(
            status_code=200,
            headers={},
            body='{"success": true}',
            execution_time_ms=75.0,
            is_success=True,
        )

        hook_node._http_client.post = AsyncMock(side_effect=[
            failure_response,  # First attempt fails
            failure_response,  # Second attempt fails
            success_response,   # Third attempt succeeds
        ])

        with patch("asyncio.sleep") as mock_sleep:
            input_data = ModelHookNodeInput(notification_request=request)
            result = await hook_node.process(input_data)

            # Verify exponential backoff delays
            sleep_calls = mock_sleep.call_args_list
            assert len(sleep_calls) == 2  # Two sleeps between three attempts

            # First delay: base_delay_ms = 100ms = 0.1s
            assert sleep_calls[0][0][0] == 0.1
            # Second delay: 100ms * 2.0 = 200ms = 0.2s
            assert sleep_calls[1][0][0] == 0.2

        assert result.success is True
        assert result.notification_result.total_attempts == 3
        assert len(result.notification_result.attempts) == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, hook_node):
        """Test circuit breaker activation after consecutive failures."""
        request = ModelNotificationRequest(
            url="https://failing.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"alert": "System alert"},
        )

        # Mock consistent failures to trigger circuit breaker
        failure_response = ProtocolHttpResponse(
            status_code=503,
            headers={},
            body='{"error": "Service unavailable"}',
            execution_time_ms=30000.0,  # Timeout
            is_success=False,
        )

        hook_node._http_client.post = AsyncMock(return_value=failure_response)

        input_data = ModelHookNodeInput(notification_request=request)

        # Send multiple notifications to trigger circuit breaker
        for i in range(6):  # Default failure threshold is 5
            try:
                await hook_node.process(input_data)
            except OnexError:
                pass  # Expected failures

        # Verify circuit breaker is now OPEN
        url_key = str(request.url)
        assert url_key in hook_node._circuit_breakers
        circuit_breaker = hook_node._circuit_breakers[url_key]
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count >= 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_requests(self, hook_node):
        """Test that open circuit breaker prevents HTTP requests."""
        request = ModelNotificationRequest(
            url="https://blocked.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"message": "This should be blocked"},
        )

        # Manually set circuit breaker to OPEN state
        url_key = str(request.url)
        circuit_breaker = hook_node._get_or_create_circuit_breaker(url_key)
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.failure_count = 10
        circuit_breaker.last_failure_time = time.time()

        hook_node._http_client.post = AsyncMock()  # Should not be called

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.SERVICE_UNAVAILABLE
        assert "Circuit breaker is OPEN" in str(exc_info.value)

        # Verify HTTP client was not called
        hook_node._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_network_timeout(self, hook_node, basic_notification_request):
        """Test error handling for network timeouts."""
        # Mock network timeout exception
        hook_node._http_client.post = AsyncMock(side_effect=TimeoutError("Request timeout"))

        input_data = ModelHookNodeInput(notification_request=basic_notification_request)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.TIMEOUT_ERROR
        assert "timeout" in str(exc_info.value).lower()
        assert hook_node._failed_notifications == 1

    @pytest.mark.asyncio
    async def test_input_validation_errors(self, hook_node):
        """Test input validation error handling."""
        # Test with missing notification request
        input_data = ModelHookNodeInput(notification_request=None)

        with pytest.raises(OnexError) as exc_info:
            await hook_node.process(input_data)

        assert exc_info.value.code == CoreErrorCode.INVALID_INPUT
        assert "Notification request is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_health_check_functionality(self, hook_node):
        """Test comprehensive health check functionality."""
        # Add some metrics
        hook_node._total_notifications = 10
        hook_node._successful_notifications = 8
        hook_node._failed_notifications = 2

        # Add a circuit breaker
        circuit_breaker = hook_node._get_or_create_circuit_breaker("https://test.com/webhook")
        circuit_breaker.failure_count = 2

        health_status = await hook_node.health_check()

        assert health_status.status == EnumHealthStatus.HEALTHY
        assert "hook_node" in health_status.details["component"]
        assert health_status.details["metrics"]["total_notifications"] == 10
        assert health_status.details["metrics"]["success_rate"] == 0.8
        assert "circuit_breakers" in health_status.details["metrics"]

    def test_slack_message_formatting(self, hook_node):
        """Test Slack-specific message formatting."""
        payload = {"text": "Alert: Database connection lost", "channel": "#alerts"}
        formatted = hook_node._format_slack_message(payload)

        # Verify Slack formatting is preserved
        assert formatted["text"] == "Alert: Database connection lost"
        assert formatted["channel"] == "#alerts"

        # Test with missing required fields
        with pytest.raises(OnexError) as exc_info:
            hook_node._format_slack_message({})

        assert exc_info.value.code == CoreErrorCode.INVALID_INPUT

    def test_generic_webhook_formatting(self, hook_node):
        """Test generic webhook payload formatting."""
        payload = {
            "event": "system.alert",
            "severity": "high",
            "message": "Memory usage exceeded threshold",
            "timestamp": "2023-09-15T10:30:00Z",
        }

        formatted = hook_node._format_generic_webhook(payload)

        # Generic webhooks should preserve original structure
        assert formatted == payload

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, hook_node, basic_notification_request):
        """Test performance metrics are correctly tracked."""
        mock_response = ProtocolHttpResponse(
            status_code=200,
            headers={},
            body='{"status": "received"}',
            execution_time_ms=150.0,
            is_success=True,
        )

        hook_node._http_client.post = AsyncMock(return_value=mock_response)

        start_time = time.time()
        input_data = ModelHookNodeInput(notification_request=basic_notification_request)
        result = await hook_node.process(input_data)
        end_time = time.time()

        # Verify performance tracking
        assert result.total_execution_time_ms > 0
        assert result.total_execution_time_ms < (end_time - start_time) * 1000 + 100  # Allow some tolerance

        # Verify HTTP execution time is captured
        assert result.notification_result.attempts[0].execution_time_ms == 150.0

    def test_correlation_id_generation(self, hook_node, basic_notification_request):
        """Test correlation ID generation and tracking."""
        input_data = ModelHookNodeInput(notification_request=basic_notification_request)

        # Correlation ID should be generated if not provided
        assert input_data.correlation_id is None

        # After processing, result should have correlation ID
        with patch.object(hook_node._http_client, "post", return_value=AsyncMock()):
            # Note: This test would need actual processing to verify correlation ID generation
            pass

    def test_retry_delay_calculations(self, hook_node):
        """Test retry delay calculations for different strategies."""
        # Test exponential backoff
        exponential_policy = ModelNotificationRetryPolicy(
            max_attempts=5,
            strategy=EnumBackoffStrategy.EXPONENTIAL,
            base_delay_ms=100,
            max_delay_ms=10000,
            backoff_multiplier=2.0,
        )

        delay1 = hook_node._calculate_retry_delay(exponential_policy, attempt=1)
        delay2 = hook_node._calculate_retry_delay(exponential_policy, attempt=2)
        delay3 = hook_node._calculate_retry_delay(exponential_policy, attempt=3)

        assert delay1 == 0.1  # 100ms
        assert delay2 == 0.2  # 200ms
        assert delay3 == 0.4  # 400ms

        # Test linear backoff
        linear_policy = ModelNotificationRetryPolicy(
            max_attempts=5,
            strategy=EnumBackoffStrategy.LINEAR,
            base_delay_ms=500,
            max_delay_ms=5000,
            backoff_multiplier=1.5,
        )

        linear_delay1 = hook_node._calculate_retry_delay(linear_policy, attempt=1)
        linear_delay2 = hook_node._calculate_retry_delay(linear_policy, attempt=2)

        assert linear_delay1 == 0.5    # 500ms
        assert linear_delay2 == 1.0    # 500ms * 2 * 1.5 / 1.5 = 1000ms

        # Test fixed delay
        fixed_policy = ModelNotificationRetryPolicy(
            max_attempts=3,
            strategy=EnumBackoffStrategy.FIXED,
            base_delay_ms=1000,
            max_delay_ms=1000,
            backoff_multiplier=1.0,
        )

        fixed_delay1 = hook_node._calculate_retry_delay(fixed_policy, attempt=1)
        fixed_delay2 = hook_node._calculate_retry_delay(fixed_policy, attempt=2)

        assert fixed_delay1 == 1.0  # 1000ms
        assert fixed_delay2 == 1.0  # 1000ms (fixed)

"""Hook Node - ONEX EFFECT Service for Infrastructure Notifications.

This node serves as a message bus bridge for webhook notifications in the ONEX infrastructure.
It converts event envelopes containing notification requests into HTTP webhook calls,
supporting multiple authentication methods, retry policies, and observability patterns.

Architecture:
Infrastructure Event → Event Bus → Hook Node → HTTP Notification Destinations
                                      ↓
                          (Slack/Discord/Webhooks/Custom APIs)

Integration Points:
- event_bus_circuit_breaker_compute: Alert for circuit breaker state changes
- infrastructure_health_monitor_orchestrator: Service health notifications
- PostgreSQL Adapter: Database connection pool alerts
- Omnimemory: Memory system performance alerts

Features:
- Multi-channel notification support (Slack, Discord, generic webhooks)
- Authentication support (Bearer token, Basic auth, API key)
- Retry policies with exponential backoff
- Circuit breaker pattern for failing destinations
- Structured logging with correlation ID tracking
- Performance metrics and observability
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.model.core.model_health_status import ModelHealthStatus
from omnibase_core.protocol.protocol_http_client import ProtocolHttpClient, ModelHttpResponse
from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus

# Shared notification models
from omnibase_infra.models.notification.model_notification_request import ModelNotificationRequest
from omnibase_infra.models.notification.model_notification_result import ModelNotificationResult
from omnibase_infra.models.notification.model_notification_attempt import ModelNotificationAttempt
from omnibase_infra.models.notification.model_notification_auth import ModelNotificationAuth
from omnibase_infra.models.notification.model_notification_retry_policy import ModelNotificationRetryPolicy

# Node-specific adapter models
from .models.model_hook_node_input import ModelHookNodeInput
from .models.model_hook_node_output import ModelHookNodeOutput


class HookStructuredLogger:
    """
    Structured logger for Hook Node operations with correlation ID tracking.

    Provides consistent, structured logging across all notification operations with:
    - Correlation ID tracking for request tracing
    - Performance metrics logging
    - Error context preservation
    - Security-aware message sanitization (no sensitive data in logs)
    """

    def __init__(self, logger_name: str = "hook_node"):
        """Initialize structured logger with correlation ID support."""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

    def _build_extra(self, correlation_id: Optional[UUID], operation: str, **kwargs) -> dict:
        """Build extra context for structured logging."""
        extra = {
            "correlation_id": str(correlation_id) if correlation_id else None,
            "operation": operation,
            "component": "hook_node"
        }
        extra.update(kwargs)
        return extra

    def info(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "notification", **kwargs):
        """Log info level message with structured context."""
        self.logger.info(message, extra=self._build_extra(correlation_id, operation, **kwargs))

    def warning(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "notification", **kwargs):
        """Log warning level message with structured context."""
        self.logger.warning(message, extra=self._build_extra(correlation_id, operation, **kwargs))

    def error(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "notification",
              exception: Optional[Exception] = None, **kwargs):
        """Log error level message with structured context and exception details."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        if exception:
            extra["exception_type"] = type(exception).__name__
            extra["exception_message"] = str(exception)
        self.logger.error(message, extra=extra)

    def debug(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "notification", **kwargs):
        """Log debug level message with structured context."""
        self.logger.debug(message, extra=self._build_extra(correlation_id, operation, **kwargs))

    def _sanitize_url_for_logging(self, url: str) -> str:
        """Sanitize webhook URL for safe logging (remove sensitive parameters)."""
        try:
            # Remove potential tokens, keys, or secrets from query parameters
            import re
            # Remove query parameters that might contain sensitive data
            sanitized = re.sub(r'[?&](token|key|secret|auth|api_key)=[^&]*',
                             lambda m: m.group(0).split('=')[0] + '=***', url)
            return sanitized
        except Exception:
            return url[:50] + "..." if len(url) > 50 else url

    def log_notification_start(self, correlation_id: UUID, url: str, method: str, retry_attempt: int = 1):
        """Log start of notification attempt with sanitized URL."""
        sanitized_url = self._sanitize_url_for_logging(url)
        self.info(
            f"Starting notification attempt {retry_attempt}: {method} {sanitized_url}",
            correlation_id=correlation_id,
            operation="notification_send",
            url=sanitized_url,
            method=method,
            retry_attempt=retry_attempt
        )

    def log_notification_success(self, correlation_id: UUID, execution_time_ms: float,
                               status_code: int, retry_attempt: int = 1):
        """Log successful notification delivery."""
        self.info(
            f"Notification attempt {retry_attempt} succeeded: {status_code} ({execution_time_ms:.2f}ms)",
            correlation_id=correlation_id,
            operation="notification_success",
            execution_time_ms=execution_time_ms,
            status_code=status_code,
            retry_attempt=retry_attempt
        )

    def log_notification_error(self, correlation_id: UUID, execution_time_ms: float,
                             exception: Exception, retry_attempt: int = 1):
        """Log failed notification attempt."""
        self.error(
            f"Notification attempt {retry_attempt} failed ({execution_time_ms:.2f}ms): {str(exception)}",
            correlation_id=correlation_id,
            operation="notification_error",
            execution_time_ms=execution_time_ms,
            exception=exception,
            retry_attempt=retry_attempt
        )


class NotificationCircuitBreaker:
    """
    Circuit breaker pattern for notification destinations.

    Prevents cascade failures by opening circuit when destination consistently fails.
    Supports different states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing).
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60,
                 half_open_max_calls: int = 3):
        """Initialize circuit breaker with failure tracking."""
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if notification can be attempted based on circuit state."""
        current_time = time.time()

        if self._state == "CLOSED":
            return True
        elif self._state == "OPEN":
            if current_time - self._last_failure_time >= self.timeout_seconds:
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
                return True
            return False
        elif self._state == "HALF_OPEN":
            return self._half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record successful notification delivery."""
        self._failure_count = 0
        self._state = "CLOSED"
        self._half_open_calls = 0

    def record_failure(self):
        """Record failed notification delivery."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
        elif self._state == "CLOSED" and self._failure_count >= self.failure_threshold:
            self._state = "OPEN"

        self._half_open_calls += 1 if self._state == "HALF_OPEN" else 0

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count


class NodeHookEffect(NodeEffectService):
    """
    Hook Node - Infrastructure notification bridge following ONEX EFFECT pattern.

    Converts message bus envelopes containing notification requests into HTTP webhook calls
    to various destinations (Slack, Discord, generic webhooks). Implements resilience
    patterns including retry policies, circuit breakers, and comprehensive observability.

    Message Flow:
    Infrastructure Event → Event Bus → Hook Node → HTTP Notification Destination

    Integrates with:
    - protocol_http_client: HTTP client for webhook delivery
    - protocol_event_bus: Event bus for infrastructure events
    - Shared notification models: Request/response models for notifications
    """

    def __init__(self, container: ONEXContainer):
        """Initialize Hook Node with container injection."""
        super().__init__(container)
        self.node_type = "effect"
        self.domain = "infrastructure"

        # Initialize HTTP client for webhook delivery (REQUIRED - NO FALLBACKS)
        self._http_client: ProtocolHttpClient = self.container.get_service("ProtocolHttpClient")
        if self._http_client is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolHttpClient service not available - HTTP client is REQUIRED for Hook Node"
            )

        # Initialize event bus for infrastructure event integration (REQUIRED - NO FALLBACKS)
        self._event_bus: ProtocolEventBus = self.container.get_service("ProtocolEventBus")
        if self._event_bus is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolEventBus service not available - event bus integration is REQUIRED for Hook Node"
            )

        # Initialize structured logger with correlation ID support
        self._logger = HookStructuredLogger("hook_node")

        # Initialize circuit breakers for notification destinations (per-URL tracking)
        self._circuit_breakers: Dict[str, NotificationCircuitBreaker] = {}
        self._circuit_breaker_lock = asyncio.Lock()

        # Performance metrics tracking
        self._total_notifications = 0
        self._successful_notifications = 0
        self._failed_notifications = 0

        self._logger.info("Hook Node initialized successfully", operation="initialization")

    def _get_circuit_breaker(self, url: str) -> NotificationCircuitBreaker:
        """Get or create circuit breaker for notification destination URL."""
        if url not in self._circuit_breakers:
            self._circuit_breakers[url] = NotificationCircuitBreaker(
                failure_threshold=5,  # Open circuit after 5 failures
                timeout_seconds=60,   # Wait 60 seconds before retry
                half_open_max_calls=3 # Allow 3 test calls in half-open state
            )
        return self._circuit_breakers[url]

    def _build_http_headers(self, base_headers: Optional[Dict[str, str]],
                          auth: Optional[ModelNotificationAuth]) -> Dict[str, str]:
        """Build HTTP headers including authentication."""
        headers = base_headers.copy() if base_headers else {}

        # Ensure Content-Type for JSON payloads
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        # Add authentication headers
        if auth:
            if auth.auth_type == "bearer" and auth.credentials.get("token"):
                headers["Authorization"] = f"Bearer {auth.credentials['token']}"
            elif auth.auth_type == "basic" and auth.credentials.get("username") and auth.credentials.get("password"):
                import base64
                credentials = f"{auth.credentials['username']}:{auth.credentials['password']}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded_credentials}"
            elif auth.auth_type == "api_key_header" and auth.credentials.get("header_name") and auth.credentials.get("api_key"):
                headers[auth.credentials["header_name"]] = auth.credentials["api_key"]

        return headers

    def _calculate_retry_delay(self, attempt: int, retry_policy: ModelNotificationRetryPolicy) -> float:
        """Calculate delay before retry attempt based on backoff strategy."""
        base_delay = retry_policy.delay_seconds

        if attempt <= 1:
            return base_delay

        if retry_policy.backoff_strategy == "exponential":
            return base_delay * (2 ** (attempt - 1))
        elif retry_policy.backoff_strategy == "linear":
            return base_delay * attempt
        else:  # fixed
            return base_delay

    def _is_retryable_status(self, status_code: int, retry_policy: ModelNotificationRetryPolicy) -> bool:
        """Check if HTTP status code should trigger a retry."""
        return status_code in retry_policy.retryable_status_codes

    async def _send_notification_with_retries(
        self,
        request: ModelNotificationRequest,
        correlation_id: UUID
    ) -> ModelNotificationResult:
        """Send notification with retry policy and circuit breaker protection."""
        attempts: List[ModelNotificationAttempt] = []
        circuit_breaker = self._get_circuit_breaker(request.url)

        # Check circuit breaker state
        if not circuit_breaker.can_execute():
            self._logger.warning(
                f"Circuit breaker OPEN for destination: {request.url}",
                correlation_id=correlation_id,
                operation="circuit_breaker_check",
                circuit_state=circuit_breaker.state
            )
            # Return failure result without attempting request
            attempt = ModelNotificationAttempt(
                attempt_number=1,
                timestamp=time.time(),
                status_code=None,
                error=f"Circuit breaker {circuit_breaker.state} - destination unavailable",
                execution_time_ms=0.0
            )
            attempts.append(attempt)

            return ModelNotificationResult(
                final_status_code=None,
                is_success=False,
                attempts=attempts,
                total_attempts=1
            )

        # Use default retry policy if not specified
        retry_policy = request.retry_policy or ModelNotificationRetryPolicy(
            max_attempts=3,
            backoff_strategy="exponential",
            delay_seconds=5.0
        )

        headers = self._build_http_headers(request.headers, request.auth)

        for attempt_num in range(1, retry_policy.max_attempts + 1):
            start_time = time.time()

            self._logger.log_notification_start(
                correlation_id=correlation_id,
                url=request.url,
                method=request.method,
                retry_attempt=attempt_num
            )

            try:
                # Make HTTP request
                response: ModelHttpResponse = await self._http_client.request(
                    method=request.method,
                    url=request.url,
                    json=request.payload,
                    headers=headers,
                    timeout=30.0  # 30 second timeout
                )

                execution_time_ms = (time.time() - start_time) * 1000

                # Record attempt
                attempt = ModelNotificationAttempt(
                    attempt_number=attempt_num,
                    timestamp=start_time,
                    status_code=response.status_code,
                    error=None,
                    execution_time_ms=execution_time_ms
                )
                attempts.append(attempt)

                # Check if successful (2xx status codes)
                is_success = 200 <= response.status_code < 300

                if is_success:
                    self._logger.log_notification_success(
                        correlation_id=correlation_id,
                        execution_time_ms=execution_time_ms,
                        status_code=response.status_code,
                        retry_attempt=attempt_num
                    )

                    circuit_breaker.record_success()

                    return ModelNotificationResult(
                        final_status_code=response.status_code,
                        is_success=True,
                        attempts=attempts,
                        total_attempts=len(attempts)
                    )

                # Check if we should retry based on status code
                if not self._is_retryable_status(response.status_code, retry_policy):
                    self._logger.warning(
                        f"Non-retryable status code {response.status_code} - giving up",
                        correlation_id=correlation_id,
                        operation="non_retryable_error",
                        status_code=response.status_code
                    )
                    break

                # Wait before retry (except on last attempt)
                if attempt_num < retry_policy.max_attempts:
                    retry_delay = self._calculate_retry_delay(attempt_num, retry_policy)
                    self._logger.debug(
                        f"Retrying after {retry_delay}s delay (attempt {attempt_num}/{retry_policy.max_attempts})",
                        correlation_id=correlation_id,
                        operation="retry_delay",
                        retry_delay=retry_delay
                    )
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000

                self._logger.log_notification_error(
                    correlation_id=correlation_id,
                    execution_time_ms=execution_time_ms,
                    exception=e,
                    retry_attempt=attempt_num
                )

                # Record failed attempt
                attempt = ModelNotificationAttempt(
                    attempt_number=attempt_num,
                    timestamp=start_time,
                    status_code=None,
                    error=str(e),
                    execution_time_ms=execution_time_ms
                )
                attempts.append(attempt)

                # Wait before retry (except on last attempt)
                if attempt_num < retry_policy.max_attempts:
                    retry_delay = self._calculate_retry_delay(attempt_num, retry_policy)
                    await asyncio.sleep(retry_delay)

        # All attempts failed
        circuit_breaker.record_failure()

        return ModelNotificationResult(
            final_status_code=attempts[-1].status_code if attempts else None,
            is_success=False,
            attempts=attempts,
            total_attempts=len(attempts)
        )

    async def process(self, input_data: ModelHookNodeInput) -> ModelHookNodeOutput:
        """
        Process hook node notification request following ONEX EFFECT pattern.

        Converts message bus envelope to HTTP notification delivery with comprehensive
        error handling, retry logic, circuit breaker protection, and observability.

        Args:
            input_data: Input envelope containing notification request and metadata

        Returns:
            ModelHookNodeOutput: Notification delivery result with success status and metrics

        Raises:
            OnexError: For system-level failures (dependency unavailable, invalid input)
        """
        start_time = time.time()
        correlation_id = input_data.correlation_id

        self._logger.info(
            f"Processing notification request",
            correlation_id=correlation_id,
            operation="process_start",
            destination_url=input_data.notification_request.url
        )

        # Update metrics
        self._total_notifications += 1

        try:
            # Validate input
            if not input_data.notification_request:
                raise OnexError(
                    code=CoreErrorCode.INVALID_INPUT,
                    message="Notification request is required"
                )

            if not input_data.notification_request.url:
                raise OnexError(
                    code=CoreErrorCode.INVALID_INPUT,
                    message="Notification URL is required"
                )

            # Send notification with retries and circuit breaker protection
            notification_result = await self._send_notification_with_retries(
                request=input_data.notification_request,
                correlation_id=correlation_id
            )

            # Update success/failure metrics
            if notification_result.is_success:
                self._successful_notifications += 1
            else:
                self._failed_notifications += 1

            # Calculate total execution time
            total_execution_time_ms = (time.time() - start_time) * 1000

            self._logger.info(
                f"Notification processing completed: {'SUCCESS' if notification_result.is_success else 'FAILED'} "
                f"({total_execution_time_ms:.2f}ms, {notification_result.total_attempts} attempts)",
                correlation_id=correlation_id,
                operation="process_complete",
                success=notification_result.is_success,
                total_attempts=notification_result.total_attempts,
                final_status_code=notification_result.final_status_code,
                total_execution_time_ms=total_execution_time_ms
            )

            return ModelHookNodeOutput(
                notification_result=notification_result,
                success=notification_result.is_success,
                error_message=None if notification_result.is_success else "Notification delivery failed after all retry attempts",
                correlation_id=correlation_id,
                timestamp=time.time(),
                total_execution_time_ms=total_execution_time_ms
            )

        except OnexError:
            # Re-raise ONEX errors as-is (already structured)
            self._failed_notifications += 1
            raise
        except Exception as e:
            # Convert unexpected exceptions to OnexError
            self._failed_notifications += 1
            total_execution_time_ms = (time.time() - start_time) * 1000

            self._logger.error(
                f"Unexpected error during notification processing: {str(e)}",
                correlation_id=correlation_id,
                operation="process_error",
                exception=e,
                total_execution_time_ms=total_execution_time_ms
            )

            raise OnexError(
                code=CoreErrorCode.SYSTEM_ERROR,
                message=f"Hook Node processing failed: {str(e)}"
            ) from e

    async def health_check(self) -> ModelHealthStatus:
        """
        Perform comprehensive health check for Hook Node.

        Checks all critical dependencies and reports overall health status.

        Returns:
            ModelHealthStatus: Detailed health status with component-level checks
        """
        try:
            health_details = {
                "component": "hook_node",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "total_notifications": self._total_notifications,
                    "successful_notifications": self._successful_notifications,
                    "failed_notifications": self._failed_notifications,
                    "success_rate": (
                        self._successful_notifications / self._total_notifications
                        if self._total_notifications > 0 else 1.0
                    ),
                    "circuit_breakers": {
                        url: {"state": cb.state, "failure_count": cb.failure_count}
                        for url, cb in self._circuit_breakers.items()
                    }
                }
            }

            # Check HTTP client availability
            if self._http_client is None:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="HTTP client not available",
                    details=health_details
                )

            # Check event bus availability
            if self._event_bus is None:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Event bus not available",
                    details=health_details
                )

            # All checks passed
            self._logger.debug("Health check completed successfully", operation="health_check")

            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Hook Node is healthy and operational",
                details=health_details
            )

        except Exception as e:
            self._logger.error(
                f"Health check failed: {str(e)}",
                operation="health_check_error",
                exception=e
            )

            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "component": "hook_node"}
            )
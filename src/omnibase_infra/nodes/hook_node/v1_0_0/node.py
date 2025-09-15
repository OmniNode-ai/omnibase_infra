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

SECURITY ENHANCEMENTS (PR #6 Critical Fixes):
✅ SSRF Prevention: Comprehensive URL validation blocking private networks, localhost, cloud metadata services
✅ Payload Size Limits: 1MB configurable limit with security violation detection
✅ Rate Limiting: Per-destination DoS protection (60 req/min default)
✅ Protocol Compliance: Using request() method per ProtocolHttpClient interface
✅ Race Condition Fix: Atomic circuit breaker state reporting with async locking
✅ Security Logging: Comprehensive security event logging and monitoring
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from enum import Enum
from ipaddress import AddressValueError, IPv4Address, IPv6Address, ip_address
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse
from uuid import UUID, uuid4

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.node import EnumHealthStatus
from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_core.enums.enum_auth_type import EnumAuthType
from omnibase_core.enums.enum_backoff_strategy import EnumBackoffStrategy
from omnibase_core.models.core.model_health_status import ModelHealthStatus
from omnibase_core.model.core.model_onex_event import ModelOnexEvent
from omnibase_spi.protocols.core import ProtocolHttpClient, ProtocolHttpResponse
from omnibase_spi.protocols.event_bus import ProtocolEventBus

# Shared notification models
from omnibase_infra.models.notification.model_notification_request import ModelNotificationRequest
from omnibase_infra.models.notification.model_notification_result import ModelNotificationResult
from omnibase_infra.models.notification.model_notification_attempt import ModelNotificationAttempt
from omnibase_infra.models.notification.model_notification_auth import ModelNotificationAuth
from omnibase_infra.models.notification.model_notification_retry_policy import ModelNotificationRetryPolicy

# Node-specific adapter models
from .models.model_hook_node_input import ModelHookNodeInput
from .models.model_hook_node_output import ModelHookNodeOutput


class CircuitBreakerState(Enum):
    """Circuit breaker states for notification destinations."""
    CLOSED = "closed"       # Normal operation - notifications sent directly
    OPEN = "open"          # Failure state - notifications blocked
    HALF_OPEN = "half_open"  # Testing state - limited notifications to test recovery


class SecurityConfig:
    """Security configuration for Hook Node operations."""

    def __init__(self):
        # SSRF Protection - RFC 1918 private networks and special addresses
        self.blocked_ip_ranges = [
            # RFC 1918 private networks
            "10.0.0.0/8",      # Class A private network
            "172.16.0.0/12",   # Class B private network
            "192.168.0.0/16",  # Class C private network
            # Localhost and loopback
            "127.0.0.0/8",     # IPv4 loopback
            "::1/128",         # IPv6 loopback
            # Link-local addresses
            "169.254.0.0/16",  # IPv4 link-local (including cloud metadata)
            "fe80::/10",       # IPv6 link-local
            # Multicast
            "224.0.0.0/4",     # IPv4 multicast
            "ff00::/8",        # IPv6 multicast
        ]

        # Cloud metadata service addresses (critical for SSRF prevention)
        self.blocked_metadata_addresses = [
            "169.254.169.254",  # AWS/GCP/Azure metadata service
            "fd00:ec2::254",    # AWS IPv6 metadata service
        ]

        # Payload size limits (1MB default, configurable)
        self.max_payload_size_bytes = 1024 * 1024  # 1MB

        # Rate limiting configuration
        self.rate_limit_requests_per_minute = 60
        self.rate_limit_window_seconds = 60


class UrlSecurityValidator:
    """URL security validator for SSRF prevention."""

    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self._compiled_ip_ranges = []
        self._compile_ip_ranges()

    def _compile_ip_ranges(self):
        """Pre-compile IP ranges for efficient validation."""
        from ipaddress import ip_network
        for range_str in self.config.blocked_ip_ranges:
            try:
                self._compiled_ip_ranges.append(ip_network(range_str, strict=False))
            except Exception as e:
                # Log but don't fail initialization for invalid ranges
                logging.warning(f"Invalid IP range in security config: {range_str}: {e}")

    def validate_url(self, url: str) -> None:
        """
        Validate URL for SSRF prevention.

        Args:
            url: URL to validate

        Raises:
            OnexError: If URL is blocked for security reasons
        """
        if not url or not url.strip():
            raise OnexError(
                code=CoreErrorCode.INVALID_INPUT,
                message="URL cannot be empty"
            )

        try:
            parsed = urlparse(url.strip())

            # Validate scheme
            if parsed.scheme not in ['http', 'https']:
                raise OnexError(
                    code=CoreErrorCode.SECURITY_VIOLATION,
                    message=f"Blocked URL scheme: {parsed.scheme}. Only http/https allowed."
                )

            # Validate hostname exists
            if not parsed.hostname:
                raise OnexError(
                    code=CoreErrorCode.SECURITY_VIOLATION,
                    message="URL must contain a valid hostname"
                )

            # Check for blocked metadata addresses first (exact match)
            if parsed.hostname in self.config.blocked_metadata_addresses:
                raise OnexError(
                    code=CoreErrorCode.SECURITY_VIOLATION,
                    message=f"Blocked metadata service address: {parsed.hostname}"
                )

            # Resolve hostname to IP and check against blocked ranges
            try:
                ip_addr = ip_address(parsed.hostname)
                self._validate_ip_address(ip_addr, parsed.hostname)
            except AddressValueError:
                # Hostname is not an IP address, resolve it
                import socket
                try:
                    # Get all IP addresses for the hostname
                    addr_info = socket.getaddrinfo(parsed.hostname, parsed.port, family=socket.AF_UNSPEC)
                    for family, type_, proto, canonname, sockaddr in addr_info:
                        ip_str = sockaddr[0]
                        try:
                            ip_addr = ip_address(ip_str)
                            self._validate_ip_address(ip_addr, f"{parsed.hostname} -> {ip_str}")
                        except AddressValueError:
                            continue  # Skip invalid IP addresses
                except socket.gaierror as e:
                    raise OnexError(
                        code=CoreErrorCode.SECURITY_VIOLATION,
                        message=f"Cannot resolve hostname {parsed.hostname}: {e}"
                    )

            # Additional hostname validation
            self._validate_hostname(parsed.hostname)

        except OnexError:
            raise  # Re-raise OnexError as-is
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.SECURITY_VIOLATION,
                message=f"URL validation failed: {e}"
            ) from e

    def _validate_ip_address(self, ip_addr: Union[IPv4Address, IPv6Address], display_name: str) -> None:
        """Validate IP address against blocked ranges."""
        for blocked_range in self._compiled_ip_ranges:
            if ip_addr in blocked_range:
                raise OnexError(
                    code=CoreErrorCode.SECURITY_VIOLATION,
                    message=f"Blocked IP address {display_name} in range {blocked_range}"
                )

    def _validate_hostname(self, hostname: str) -> None:
        """Additional hostname validation."""
        # Check for localhost variants
        localhost_patterns = [
            'localhost', '0.0.0.0', '0', 'local', 'localdomain'
        ]
        if hostname.lower() in localhost_patterns:
            raise OnexError(
                code=CoreErrorCode.SECURITY_VIOLATION,
                message=f"Blocked localhost hostname: {hostname}"
            )


class RateLimiter:
    """Rate limiter for notification destinations."""

    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, destination_url: str) -> bool:
        """
        Check if request is within rate limit for destination URL.

        Args:
            destination_url: Destination URL for rate limiting

        Returns:
            bool: True if within rate limit, False if rate limited
        """
        current_time = time.time()

        async with self._lock:
            # Clean up old requests outside the window
            if destination_url in self._requests:
                cutoff_time = current_time - self.window_seconds
                self._requests[destination_url] = [
                    req_time for req_time in self._requests[destination_url]
                    if req_time > cutoff_time
                ]
            else:
                self._requests[destination_url] = []

            # Check if adding this request would exceed the limit
            if len(self._requests[destination_url]) >= self.requests_per_minute:
                return False

            # Add the current request
            self._requests[destination_url].append(current_time)
            return True

    async def get_rate_limit_status(self, destination_url: str) -> Dict[str, Union[int, float]]:
        """Get current rate limit status for a destination."""
        current_time = time.time()

        async with self._lock:
            if destination_url not in self._requests:
                return {
                    "current_requests": 0,
                    "limit": self.requests_per_minute,
                    "window_seconds": self.window_seconds,
                    "remaining": self.requests_per_minute
                }

            # Clean up old requests
            cutoff_time = current_time - self.window_seconds
            active_requests = [
                req_time for req_time in self._requests[destination_url]
                if req_time > cutoff_time
            ]

            return {
                "current_requests": len(active_requests),
                "limit": self.requests_per_minute,
                "window_seconds": self.window_seconds,
                "remaining": max(0, self.requests_per_minute - len(active_requests))
            }


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

    def _build_extra(self, correlation_id: Optional[str], operation: str, **kwargs) -> dict:
        """Build extra context for structured logging."""
        extra = {
            "correlation_id": correlation_id,
            "operation": operation,
            "component": "hook_node"
        }
        extra.update(kwargs)
        return extra

    def info(self, message: str, correlation_id: Optional[str] = None, operation: str = "notification", **kwargs):
        """Log info level message with structured context."""
        self.logger.info(message, extra=self._build_extra(correlation_id, operation, **kwargs))

    def warning(self, message: str, correlation_id: Optional[str] = None, operation: str = "notification", **kwargs):
        """Log warning level message with structured context."""
        self.logger.warning(message, extra=self._build_extra(correlation_id, operation, **kwargs))

    def error(self, message: str, correlation_id: Optional[str] = None, operation: str = "notification",
              exception: Optional[Exception] = None, **kwargs):
        """Log error level message with structured context and exception details."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        if exception:
            extra["exception_type"] = type(exception).__name__
            extra["exception_message"] = str(exception)
        self.logger.error(message, extra=extra)

    def debug(self, message: str, correlation_id: Optional[str] = None, operation: str = "notification", **kwargs):
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

    def log_notification_start(self, correlation_id: str, url: str, method: str, retry_attempt: int = 1):
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

    def log_notification_success(self, correlation_id: str, execution_time_ms: float,
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

    def log_notification_error(self, correlation_id: str, execution_time_ms: float,
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

    Thread-safe with async locking to prevent race conditions in concurrent environments.
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60,
                 half_open_max_calls: int = 3):
        """Initialize circuit breaker with failure tracking."""
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time = 0
        self._state = CircuitBreakerState.CLOSED
        self._half_open_calls = 0
        self._lock = asyncio.Lock()  # Async lock for thread safety

    async def can_execute(self) -> bool:
        """Check if notification can be attempted based on circuit state."""
        async with self._lock:
            current_time = time.time()

            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                if current_time - self._last_failure_time >= self.timeout_seconds:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    return True
                return False
            elif self._state == CircuitBreakerState.HALF_OPEN:
                return self._half_open_calls < self.half_open_max_calls

            return False

    async def record_success(self) -> tuple[bool, CircuitBreakerState, CircuitBreakerState, int]:
        """
        Record successful notification delivery.

        Returns:
            tuple: (state_changed, old_state, new_state, failure_count)
        """
        async with self._lock:
            old_state = self._state
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED
            self._half_open_calls = 0
            new_state = self._state
            return (old_state != new_state, old_state, new_state, self._failure_count)

    async def record_failure(self) -> tuple[bool, CircuitBreakerState, CircuitBreakerState, int]:
        """
        Record failed notification delivery.

        Returns:
            tuple: (state_changed, old_state, new_state, failure_count)
        """
        async with self._lock:
            old_state = self._state
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
            elif self._state == CircuitBreakerState.CLOSED and self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN

            self._half_open_calls += 1 if self._state == CircuitBreakerState.HALF_OPEN else 0
            new_state = self._state
            return (old_state != new_state, old_state, new_state, self._failure_count)

    async def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        async with self._lock:
            return self._state

    async def get_failure_count(self) -> int:
        """Get current failure count."""
        async with self._lock:
            return self._failure_count

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state (non-async for backward compatibility)."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count (non-async for backward compatibility)."""
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

    def __init__(self, container: ModelONEXContainer):
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

        # Initialize security components (CRITICAL - BLOCKING SECURITY VULNERABILITIES)
        self._security_config = SecurityConfig()
        self._url_validator = UrlSecurityValidator(self._security_config)
        self._rate_limiter = RateLimiter(
            requests_per_minute=self._security_config.rate_limit_requests_per_minute,
            window_seconds=self._security_config.rate_limit_window_seconds
        )

        # Initialize bounded circuit breakers for notification destinations (per-URL tracking)
        self._circuit_breakers: Dict[str, NotificationCircuitBreaker] = {}
        self._circuit_breaker_access_order: List[str] = []  # LRU tracking for bounded storage
        self._max_circuit_breakers = 1000  # Maximum number of circuit breakers to store
        self._circuit_breaker_lock = asyncio.Lock()  # Global lock for circuit breaker dict management

        # Performance metrics tracking
        self._total_notifications = 0
        self._successful_notifications = 0
        self._failed_notifications = 0

        self._logger.info(
            "Hook Node initialized successfully with security protections",
            operation="initialization",
            security_config={
                "max_payload_size_mb": self._security_config.max_payload_size_bytes / (1024 * 1024),
                "rate_limit_per_minute": self._security_config.rate_limit_requests_per_minute,
                "blocked_ip_ranges_count": len(self._security_config.blocked_ip_ranges),
                "ssrf_protection_enabled": True
            }
        )

    async def _get_circuit_breaker(self, url: str) -> NotificationCircuitBreaker:
        """
        Get or create circuit breaker for notification destination URL.

        Implements LRU (Least Recently Used) bounded storage to prevent memory leaks.
        When the maximum number of circuit breakers is reached, the least recently
        used circuit breaker is removed.
        """
        async with self._circuit_breaker_lock:
            # If circuit breaker exists, move to end of access order (most recently used)
            if url in self._circuit_breakers:
                self._circuit_breaker_access_order.remove(url)
                self._circuit_breaker_access_order.append(url)
                return self._circuit_breakers[url]

            # If we're at capacity, remove the least recently used circuit breaker
            if len(self._circuit_breakers) >= self._max_circuit_breakers:
                oldest_url = self._circuit_breaker_access_order.pop(0)
                del self._circuit_breakers[oldest_url]
                self._logger.debug(
                    f"Removed LRU circuit breaker for {oldest_url} (capacity limit: {self._max_circuit_breakers})",
                    operation="circuit_breaker_lru_eviction",
                    evicted_url=oldest_url,
                    total_circuit_breakers=len(self._circuit_breakers)
                )

            # Create new circuit breaker
            self._circuit_breakers[url] = NotificationCircuitBreaker(
                failure_threshold=5,  # Open circuit after 5 failures
                timeout_seconds=60,   # Wait 60 seconds before retry
                half_open_max_calls=3 # Allow 3 test calls in half-open state
            )
            self._circuit_breaker_access_order.append(url)
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
            if auth.auth_type == EnumAuthType.BEARER and auth.credentials.get("token"):
                headers["Authorization"] = f"Bearer {auth.credentials['token']}"
            elif auth.auth_type == EnumAuthType.BASIC and auth.credentials.get("username") and auth.credentials.get("password"):
                import base64
                credentials = f"{auth.credentials['username']}:{auth.credentials['password']}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded_credentials}"
            elif auth.auth_type == EnumAuthType.API_KEY_HEADER and auth.credentials.get("header_name") and auth.credentials.get("api_key"):
                headers[auth.credentials["header_name"]] = auth.credentials["api_key"]

        return headers

    def _validate_payload_size(self, payload: Dict) -> None:
        """
        Validate payload size against security limits.

        Args:
            payload: JSON payload to validate

        Raises:
            OnexError: If payload exceeds size limits
        """
        try:
            # Calculate payload size by serializing to JSON
            payload_json = json.dumps(payload, separators=(',', ':'))  # Compact JSON
            payload_size = len(payload_json.encode('utf-8'))

            if payload_size > self._security_config.max_payload_size_bytes:
                raise OnexError(
                    code=CoreErrorCode.SECURITY_VIOLATION,
                    message=f"Payload size {payload_size} bytes exceeds maximum allowed "
                           f"{self._security_config.max_payload_size_bytes} bytes "
                           f"({self._security_config.max_payload_size_bytes / (1024*1024):.1f}MB)"
                )

            self._logger.debug(
                f"Payload size validation passed: {payload_size} bytes",
                operation="payload_validation",
                payload_size_bytes=payload_size,
                max_allowed_bytes=self._security_config.max_payload_size_bytes
            )

        except OnexError:
            raise  # Re-raise OnexError as-is
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.SECURITY_VIOLATION,
                message=f"Payload size validation failed: {e}"
            ) from e

    def _calculate_retry_delay(self, attempt: int, retry_policy: ModelNotificationRetryPolicy) -> float:
        """Calculate delay before retry attempt based on backoff strategy."""
        base_delay = retry_policy.delay_seconds

        if attempt <= 1:
            return base_delay

        if retry_policy.backoff_strategy == EnumBackoffStrategy.EXPONENTIAL:
            return base_delay * (2 ** (attempt - 1))
        elif retry_policy.backoff_strategy == EnumBackoffStrategy.LINEAR:
            return base_delay * attempt
        else:  # fixed or unknown - default to fixed
            return base_delay

    def _is_retryable_status(self, status_code: int, retry_policy: ModelNotificationRetryPolicy) -> bool:
        """Check if HTTP status code should trigger a retry."""
        return status_code in retry_policy.retryable_status_codes

    async def _publish_circuit_breaker_success_event(
        self,
        correlation_id: str,
        destination_url: str,
        status_code: int,
        execution_time_ms: float
    ) -> None:
        """Publish circuit breaker success event to event bus."""
        try:
            event = ModelOnexEvent(
                event_type="circuit_breaker.success",
                source="hook_node",
                correlation_id=correlation_id,
                timestamp=datetime.utcnow().isoformat(),
                payload={
                    "destination_url": destination_url,
                    "status_code": status_code,
                    "execution_time_ms": execution_time_ms,
                    "circuit_breaker_state": "success_recorded"
                }
            )
            await self._event_bus.publish_event(event)
        except Exception as e:
            self._logger.warning(
                f"Failed to publish circuit breaker success event: {e}",
                correlation_id=correlation_id,
                operation="event_publishing"
            )

    async def _publish_circuit_breaker_failure_event(
        self,
        correlation_id: str,
        destination_url: str,
        error_message: str,
        execution_time_ms: float
    ) -> None:
        """Publish circuit breaker failure event to event bus."""
        try:
            event = ModelOnexEvent(
                event_type="circuit_breaker.failure",
                source="hook_node",
                correlation_id=correlation_id,
                timestamp=datetime.utcnow().isoformat(),
                payload={
                    "destination_url": destination_url,
                    "error_message": error_message,
                    "execution_time_ms": execution_time_ms,
                    "circuit_breaker_state": "failure_recorded"
                }
            )
            await self._event_bus.publish_event(event)
        except Exception as e:
            self._logger.warning(
                f"Failed to publish circuit breaker failure event: {e}",
                correlation_id=correlation_id,
                operation="event_publishing"
            )

    async def _publish_circuit_breaker_state_change_event(
        self,
        correlation_id: str,
        destination_url: str,
        old_state: CircuitBreakerState,
        new_state: CircuitBreakerState,
        failure_count: int
    ) -> None:
        """Publish circuit breaker state change event to event bus."""
        try:
            event = ModelOnexEvent(
                event_type="circuit_breaker.state_change",
                source="hook_node",
                correlation_id=correlation_id,
                timestamp=datetime.utcnow().isoformat(),
                payload={
                    "destination_url": destination_url,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "failure_count": failure_count,
                    "state_change_reason": f"transition from {old_state.value} to {new_state.value}"
                }
            )
            await self._event_bus.publish_event(event)
        except Exception as e:
            self._logger.warning(
                f"Failed to publish circuit breaker state change event: {e}",
                correlation_id=correlation_id,
                operation="event_publishing"
            )

    async def _send_notification_with_retries(
        self,
        request: ModelNotificationRequest,
        correlation_id: str
    ) -> ModelNotificationResult:
        """Send notification with retry policy, circuit breaker, and security protections."""
        attempts: List[ModelNotificationAttempt] = []

        # CRITICAL SECURITY VALIDATION - SSRF Prevention
        try:
            self._url_validator.validate_url(request.url)
        except OnexError as e:
            self._logger.error(
                f"URL validation failed for security reasons: {e.message}",
                correlation_id=correlation_id,
                operation="url_security_validation",
                url=request.url
            )
            # Return failure result for security violation
            attempt = ModelNotificationAttempt(
                attempt_number=1,
                timestamp=time.time(),
                status_code=None,
                error=f"Security violation: {e.message}",
                execution_time_ms=0.0
            )
            attempts.append(attempt)
            return ModelNotificationResult(
                final_status_code=None,
                is_success=False,
                attempts=attempts,
                total_attempts=1
            )

        # CRITICAL SECURITY VALIDATION - Payload Size Limits
        try:
            self._validate_payload_size(request.payload)
        except OnexError as e:
            self._logger.error(
                f"Payload size validation failed: {e.message}",
                correlation_id=correlation_id,
                operation="payload_size_validation"
            )
            # Return failure result for payload size violation
            attempt = ModelNotificationAttempt(
                attempt_number=1,
                timestamp=time.time(),
                status_code=None,
                error=f"Payload size violation: {e.message}",
                execution_time_ms=0.0
            )
            attempts.append(attempt)
            return ModelNotificationResult(
                final_status_code=None,
                is_success=False,
                attempts=attempts,
                total_attempts=1
            )

        # CRITICAL SECURITY VALIDATION - Rate Limiting
        if not await self._rate_limiter.check_rate_limit(request.url):
            rate_limit_status = await self._rate_limiter.get_rate_limit_status(request.url)
            self._logger.warning(
                f"Rate limit exceeded for destination: {request.url}",
                correlation_id=correlation_id,
                operation="rate_limit_check",
                rate_limit_status=rate_limit_status
            )
            # Return failure result for rate limit violation
            attempt = ModelNotificationAttempt(
                attempt_number=1,
                timestamp=time.time(),
                status_code=None,
                error=f"Rate limit exceeded: {rate_limit_status['current_requests']}/{rate_limit_status['limit']} requests per {rate_limit_status['window_seconds']}s",
                execution_time_ms=0.0
            )
            attempts.append(attempt)
            return ModelNotificationResult(
                final_status_code=None,
                is_success=False,
                attempts=attempts,
                total_attempts=1
            )

        circuit_breaker = await self._get_circuit_breaker(request.url)

        # Check circuit breaker state
        if not await circuit_breaker.can_execute():
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
                response: ProtocolHttpResponse = await self._http_client.request(
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

                    # FIXED RACE CONDITION - Atomic circuit breaker state reporting
                    state_changed, old_state, new_state, failure_count = await circuit_breaker.record_success()

                    # Publish circuit breaker success event
                    await self._publish_circuit_breaker_success_event(
                        correlation_id=correlation_id,
                        destination_url=str(request.url),
                        status_code=response.status_code,
                        execution_time_ms=execution_time_ms
                    )

                    # Publish state change event if state changed
                    if state_changed:
                        await self._publish_circuit_breaker_state_change_event(
                            correlation_id=correlation_id,
                            destination_url=str(request.url),
                            old_state=old_state,
                            new_state=new_state,
                            failure_count=failure_count
                        )

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

        # All attempts failed - FIXED RACE CONDITION - Atomic circuit breaker state reporting
        state_changed, old_state, new_state, failure_count = await circuit_breaker.record_failure()

        # Publish circuit breaker failure event
        error_message = attempts[-1].error if attempts and attempts[-1].error else "All attempts failed"
        await self._publish_circuit_breaker_failure_event(
            correlation_id=correlation_id,
            destination_url=str(request.url),
            error_message=error_message,
            execution_time_ms=sum(attempt.execution_time_ms for attempt in attempts)
        )

        # Publish state change event if state changed
        if state_changed:
            await self._publish_circuit_breaker_state_change_event(
                correlation_id=correlation_id,
                destination_url=str(request.url),
                old_state=old_state,
                new_state=new_state,
                failure_count=failure_count
            )

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

            # Early security validation for immediate feedback
            try:
                self._url_validator.validate_url(input_data.notification_request.url)
            except OnexError as e:
                self._logger.error(
                    f"Input validation failed - URL security violation: {e.message}",
                    correlation_id=correlation_id,
                    operation="input_validation",
                    url=input_data.notification_request.url
                )
                raise  # Re-raise security violations immediately

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
            # Safely capture circuit breaker states with locking for thread-safe reporting
            async with self._circuit_breaker_lock:
                circuit_breaker_info = {}
                for url, cb in self._circuit_breakers.items():
                    # Get atomic state and failure count to prevent race conditions
                    state = await cb.get_state()
                    failure_count = await cb.get_failure_count()
                    circuit_breaker_info[url] = {
                        "state": state.value,  # Convert enum to string
                        "failure_count": failure_count
                    }

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
                    "circuit_breakers": circuit_breaker_info,
                    "circuit_breaker_storage": {
                        "current_count": len(circuit_breaker_info),
                        "max_capacity": self._max_circuit_breakers,
                        "utilization_percentage": round(len(circuit_breaker_info) / self._max_circuit_breakers * 100, 2)
                    }
                },
                "security": {
                    "ssrf_protection_enabled": True,
                    "max_payload_size_mb": self._security_config.max_payload_size_bytes / (1024 * 1024),
                    "rate_limit_requests_per_minute": self._security_config.rate_limit_requests_per_minute,
                    "blocked_ip_ranges_count": len(self._security_config.blocked_ip_ranges),
                    "blocked_metadata_addresses_count": len(self._security_config.blocked_metadata_addresses),
                    "url_validation_enabled": True
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
#!/usr/bin/env python3
"""
Circuit Breaker Configuration for NodeCodegenOrchestrator.

Provides circuit breaker patterns for external service calls (especially OnexTree intelligence).

ONEX v2.0 Compliance:
- Graceful degradation patterns
- Structured error handling
- Performance tracking
"""

import asyncio
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from circuitbreaker import CircuitBreakerError, circuit
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .models import EnumErrorCode

T = TypeVar("T")


# =============================================================================
# Circuit Breaker Decorators
# =============================================================================


def intelligence_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type[Exception] = Exception,
) -> Callable:
    """
    Circuit breaker decorator for intelligence service calls.

    Args:
        failure_threshold: Number of failures before opening circuit (default: 5)
        recovery_timeout: Seconds before attempting recovery (default: 60)
        expected_exception: Exception type to trigger circuit breaker

    Returns:
        Decorated function with circuit breaker protection
    """
    return circuit(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        name="intelligence_service_circuit",
    )


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: int = 2,
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Retries on retryable errors (network, timeout, 5xx).
    Does NOT retry on validation errors (4xx).

    Args:
        max_attempts: Maximum retry attempts (default: 3)
        initial_wait: Initial wait time in seconds (default: 1.0)
        max_wait: Maximum wait time in seconds (default: 10.0)
        exponential_base: Exponential backoff multiplier (default: 2)

    Returns:
        Decorated function with retry logic
    """

    def is_retryable_error(exception: Exception) -> bool:
        """Check if error should be retried."""
        # Network/timeout errors - always retry
        retryable_types = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            CircuitBreakerError,
        )
        if isinstance(exception, retryable_types):
            return True

        # HTTP errors - check if 5xx
        if hasattr(exception, "status_code"):
            status_code = exception.status_code
            if isinstance(status_code, int) and 500 <= status_code < 600:
                return True

        # Validation errors (4xx) - don't retry
        if hasattr(exception, "status_code"):
            status_code = exception.status_code
            if isinstance(status_code, int) and 400 <= status_code < 500:
                return False

        return False

    return retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=initial_wait,
            max=max_wait,
            exp_base=exponential_base,
        ),
        reraise=True,
    )


# =============================================================================
# Graceful Degradation Helpers
# =============================================================================


async def with_graceful_degradation(
    coro: Callable[..., Any],
    fallback_value: Any,
    error_code: EnumErrorCode,
    context: dict[str, Any],
    log_prefix: str = "Operation",
) -> tuple[Any, Optional[EnumErrorCode]]:
    """
    Execute coroutine with graceful degradation fallback.

    If the operation fails (circuit breaker open, timeout, etc.), returns
    fallback value and error code instead of raising exception.

    Args:
        coro: Coroutine to execute
        fallback_value: Value to return on failure
        error_code: Error code to return on failure
        context: Context for logging
        log_prefix: Prefix for log messages

    Returns:
        Tuple of (result or fallback_value, error_code or None)
    """
    try:
        result = await coro()
        return result, None
    except CircuitBreakerError as e:
        emit_log_event(
            LogLevel.WARNING,
            f"{log_prefix} circuit breaker open - using fallback",
            {
                **context,
                "error_code": EnumErrorCode.INTELLIGENCE_CIRCUIT_OPEN.value,
                "recovery_hint": error_code.get_recovery_hint(),
            },
        )
        return fallback_value, EnumErrorCode.INTELLIGENCE_CIRCUIT_OPEN
    except TimeoutError as e:
        emit_log_event(
            LogLevel.WARNING,
            f"{log_prefix} timed out - using fallback",
            {
                **context,
                "error_code": EnumErrorCode.INTELLIGENCE_TIMEOUT.value,
                "recovery_hint": error_code.get_recovery_hint(),
            },
        )
        return fallback_value, EnumErrorCode.INTELLIGENCE_TIMEOUT
    except RetryError as e:
        emit_log_event(
            LogLevel.WARNING,
            f"{log_prefix} failed after retries - using fallback",
            {
                **context,
                "error_code": error_code.value,
                "recovery_hint": error_code.get_recovery_hint(),
                "attempts": (
                    e.last_attempt.attempt_number
                    if hasattr(e, "last_attempt") and e.last_attempt
                    else 0
                ),
            },
        )
        return fallback_value, error_code
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"{log_prefix} failed with unexpected error - using fallback",
            {
                **context,
                "error_code": error_code.value,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "recovery_hint": error_code.get_recovery_hint(),
            },
        )
        return fallback_value, error_code


# =============================================================================
# Partial Success Handling
# =============================================================================


class PartialSuccessResult:
    """
    Result wrapper for operations that support partial success.

    Allows tracking partial results with warning flags.
    """

    def __init__(
        self,
        data: Any,
        success: bool = True,
        partial: bool = False,
        error_code: Optional[EnumErrorCode] = None,
        warnings: Optional[list[str]] = None,
    ):
        """
        Initialize partial success result.

        Args:
            data: Result data (may be partial)
            success: Whether operation fully succeeded
            partial: Whether result is partial
            error_code: Error code if partial/failed
            warnings: List of warnings
        """
        self.data = data
        self.success = success
        self.partial = partial
        self.error_code = error_code
        self.warnings = warnings or []

    def is_usable(self) -> bool:
        """Check if result is usable (full or partial success)."""
        return self.success or (self.partial and self.data is not None)

    def get_status(self) -> str:
        """Get human-readable status."""
        if self.success:
            return "SUCCESS"
        elif self.partial:
            return "PARTIAL_SUCCESS"
        else:
            return "FAILED"

    def add_warning(self, warning: str) -> None:
        """Add warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "success": self.success,
            "partial": self.partial,
            "error_code": self.error_code.value if self.error_code else None,
            "warnings": self.warnings,
            "status": self.get_status(),
        }


# =============================================================================
# Utility Functions
# =============================================================================


def is_retryable_error_code(error_code: EnumErrorCode) -> bool:
    """Check if error code represents retryable error."""
    return error_code.is_retryable


def should_use_circuit_breaker(error_code: EnumErrorCode) -> bool:
    """Check if error code should trigger circuit breaker."""
    return error_code.requires_circuit_breaker


def allows_partial_success(error_code: EnumErrorCode) -> bool:
    """Check if error code allows partial success."""
    return error_code.allows_partial_success

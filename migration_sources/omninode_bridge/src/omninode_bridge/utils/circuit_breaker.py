"""
Circuit Breaker Implementation

Provides a circuit breaker pattern implementation for handling retries,
failures, and graceful degradation in distributed systems.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeVar

from ..config.registry_config import get_registry_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker operations."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100.0


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: list[type] = field(
        default_factory=lambda: [
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ]
    )


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open or call fails after retries."""

    def __init__(
        self, message: str, state: CircuitState, metrics: CircuitBreakerMetrics
    ):
        super().__init__(message)
        self.state = state
        self.metrics = metrics


class CircuitBreaker:
    """
    Circuit breaker implementation with retry logic and metrics.

    Provides:
    - Automatic circuit opening/closing based on failure thresholds
    - Exponential backoff retry logic with jitter
    - Comprehensive metrics tracking
    - Thread-safe state management
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 60.0,
        retryable_exceptions: Optional[list[type]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name for logging and metrics
            failure_threshold: Number of consecutive failures before opening circuit
            timeout_seconds: Time to keep circuit open before attempting recovery
            max_attempts: Maximum number of retry attempts
            base_delay_seconds: Base delay for exponential backoff
            max_delay_seconds: Maximum delay between retries
            retryable_exceptions: List of exception types that should trigger retries
        """
        self.name = name
        self._failure_threshold = failure_threshold
        self._timeout_seconds = timeout_seconds

        # State management
        self._state = CircuitState.CLOSED
        self._state_lock = asyncio.Lock()
        self._last_state_change = time.time()

        # Retry configuration
        self._retry_config = RetryConfig(
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            retryable_exceptions=retryable_exceptions
            or [
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
            ],
        )

        # Metrics
        self._metrics = CircuitBreakerMetrics()

        logger.info(
            f"Circuit breaker '{name}' initialized with threshold={failure_threshold}, timeout={timeout_seconds}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self._metrics

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection and retry logic.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result if successful

        Raises:
            CircuitBreakerError: If circuit is open or all retries fail
            Exception: Original exception if not retryable or retries exhausted
        """
        async with self._state_lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._last_state_change = time.time()
                    logger.info(
                        f"Circuit breaker '{self.name}' transitioning to HALF_OPEN"
                    )
                else:
                    self._metrics.total_calls += 1
                    logger.warning(
                        f"Circuit breaker '{self.name}' is OPEN - call rejected"
                    )
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is open",
                        self._state,
                        self._metrics,
                    )

        # Execute with retry logic
        return await self._execute_with_retry(func, *args, **kwargs)

    async def _execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self._retry_config.max_attempts):
            self._metrics.total_calls += 1

            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.debug(
                        f"Circuit breaker '{self.name}' retry {attempt + 1}/{self._retry_config.max_attempts} after {delay:.2f}s delay"
                    )
                    await asyncio.sleep(delay)

                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=30.0)
                else:
                    result = func(*args, **kwargs)

                # Success - update metrics and state
                await self._record_success()
                return result

            except Exception as e:
                last_exception = e
                is_retryable = any(
                    isinstance(e, exc_type)
                    for exc_type in self._retry_config.retryable_exceptions
                )

                if is_retryable and attempt < self._retry_config.max_attempts - 1:
                    await self._record_failure(e)
                    logger.debug(
                        f"Circuit breaker '{self.name}' attempt {attempt + 1} failed with retryable exception: {e}"
                    )
                    continue
                else:
                    await self._record_failure(e)
                    if not is_retryable:
                        logger.debug(
                            f"Circuit breaker '{self.name}' attempt {attempt + 1} failed with non-retryable exception: {e}"
                        )
                    break

        # All retries failed
        logger.error(
            f"Circuit breaker '{self.name}' all {self._retry_config.max_attempts} attempts failed"
        )
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(
            self._retry_config.base_delay_seconds
            * (self._retry_config.exponential_base**attempt),
            self._retry_config.max_delay_seconds,
        )

        if self._retry_config.jitter:
            # Add jitter to prevent thundering herd
            import random

            jitter_factor = random.uniform(0.1, 0.3)
            delay = delay * (1 + jitter_factor)

        return delay

    async def _record_success(self) -> None:
        """Record successful call and update state."""
        async with self._state_lock:
            self._metrics.successful_calls += 1
            self._metrics.consecutive_failures = 0
            self._metrics.consecutive_successes += 1
            self._metrics.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Reset to closed after first success in half-open state
                self._state = CircuitState.CLOSED
                self._last_state_change = time.time()
                self._metrics.circuit_closes += 1
                logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")

    async def _record_failure(self, exception: Exception) -> None:
        """Record failed call and update state."""
        async with self._state_lock:
            self._metrics.failed_calls += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = time.time()

            if isinstance(exception, asyncio.TimeoutError):
                self._metrics.timeouts += 1

            if self._state == CircuitState.CLOSED:
                if self._metrics.consecutive_failures >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = time.time()
                    self._metrics.circuit_opens += 1
                    logger.warning(
                        f"Circuit breaker '{self.name}' transitioning to OPEN after {self._failure_threshold} consecutive failures"
                    )
            elif self._state == CircuitState.HALF_OPEN:
                # Immediate transition back to open on failure in half-open state
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
                self._metrics.circuit_opens += 1
                logger.warning(
                    f"Circuit breaker '{self.name}' transitioning back to OPEN after failure in HALF_OPEN state"
                )

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from open to half-open."""
        return time.time() - self._last_state_change >= self._timeout_seconds

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._state_lock:
            self._state = CircuitState.CLOSED
            self._last_state_change = time.time()
            self._metrics.consecutive_failures = 0
            self._metrics.consecutive_successes = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "last_state_change": self._last_state_change,
            "metrics": {
                "total_calls": self._metrics.total_calls,
                "successful_calls": self._metrics.successful_calls,
                "failed_calls": self._metrics.failed_calls,
                "timeouts": self._metrics.timeouts,
                "failure_rate": round(self._metrics.failure_rate, 2),
                "success_rate": round(self._metrics.success_rate, 2),
                "consecutive_failures": self._metrics.consecutive_failures,
                "consecutive_successes": self._metrics.consecutive_successes,
                "circuit_opens": self._metrics.circuit_opens,
                "circuit_closes": self._metrics.circuit_closes,
                "last_failure_time": self._metrics.last_failure_time,
                "last_success_time": self._metrics.last_success_time,
            },
            "config": {
                "failure_threshold": self._failure_threshold,
                "timeout_seconds": self._timeout_seconds,
                "max_attempts": self._retry_config.max_attempts,
                "base_delay_seconds": self._retry_config.base_delay_seconds,
                "max_delay_seconds": self._retry_config.max_delay_seconds,
            },
        }


# Factory function for creating circuit breakers with registry config
def create_circuit_breaker(
    name: str,
    environment: str = "development",
    config_override: Optional[dict[str, Any]] = None,
) -> CircuitBreaker:
    """
    Create circuit breaker with registry configuration.

    Args:
        name: Circuit breaker name
        environment: Environment name
        config_override: Optional configuration overrides

    Returns:
        Configured CircuitBreaker instance
    """
    registry_config = get_registry_config(environment)

    # Extract configuration from registry config
    failure_threshold = registry_config.circuit_breaker_threshold
    timeout_seconds = registry_config.circuit_breaker_timeout_seconds
    max_attempts = registry_config.max_retry_attempts
    base_delay_seconds = registry_config.retry_backoff_base_seconds
    max_delay_seconds = registry_config.retry_backoff_max_seconds

    # Apply overrides if provided
    if config_override:
        failure_threshold = config_override.get("failure_threshold", failure_threshold)
        timeout_seconds = config_override.get("timeout_seconds", timeout_seconds)
        max_attempts = config_override.get("max_attempts", max_attempts)
        base_delay_seconds = config_override.get(
            "base_delay_seconds", base_delay_seconds
        )
        max_delay_seconds = config_override.get("max_delay_seconds", max_delay_seconds)

    return CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
    )

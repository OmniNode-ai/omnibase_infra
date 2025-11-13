"""Circuit breaker implementation for service clients.

Implements the circuit breaker pattern to prevent cascading failures
in distributed systems by monitoring service health and automatically
failing fast when services are unavailable.

States:
    CLOSED: Normal operation, requests allowed
    OPEN: Service failing, requests rejected immediately
    HALF_OPEN: Testing service recovery, limited requests allowed
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: int = 60  # Seconds before attempting recovery
    half_open_max_calls: int = 3  # Max calls in half-open state
    success_threshold: int = 2  # Successes needed to close circuit
    timeout: float = 30.0  # Operation timeout in seconds
    slow_call_threshold: float = 5.0  # Seconds to consider call slow
    minimum_throughput: int = 5  # Min calls before evaluating error rate


@dataclass
class CircuitBreakerMetrics:
    """Metrics for monitoring circuit breaker behavior."""

    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    state_change_time: float = field(default_factory=time.time)
    slow_calls: int = 0
    half_open_calls: int = 0
    response_times: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        """Calculate current error rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times[-100:]) / len(self.response_times[-100:])

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "error_rate": self.error_rate,
            "avg_response_time": self.avg_response_time,
            "slow_calls": self.slow_calls,
            "last_failure_time": self.last_failure_time,
        }


class CircuitBreaker:
    """Async circuit breaker for resilient service calls.

    Example:
        cb = CircuitBreaker("my-service", config)

        async with cb.protect():
            result = await call_external_service()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            config: Configuration parameters, uses defaults if None
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics(name=name)
        self._lock = asyncio.Lock()
        self._half_open_in_flight = (
            0  # Separate counter for in-flight calls in HALF_OPEN
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self.metrics.state

    async def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.metrics.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.metrics.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    async def _transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN state."""
        async with self._lock:
            if self.metrics.state == CircuitState.OPEN:
                logger.info(
                    f"Circuit breaker '{self.name}' transitioning to HALF_OPEN state"
                )
                self.metrics.state = CircuitState.HALF_OPEN
                self.metrics.state_change_time = time.time()
                self.metrics.half_open_calls = 0
                self.metrics.consecutive_successes = 0
                self._half_open_in_flight = 0  # Reset in-flight counter

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state due to failures."""
        async with self._lock:
            if self.metrics.state != CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker '{self.name}' opening due to failures "
                    f"(consecutive: {self.metrics.consecutive_failures}, "
                    f"error_rate: {self.metrics.error_rate:.2%})"
                )
                self.metrics.state = CircuitState.OPEN
                self.metrics.state_change_time = time.time()

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state after recovery."""
        async with self._lock:
            if self.metrics.state != CircuitState.CLOSED:
                logger.info(
                    f"Circuit breaker '{self.name}' closing after successful recovery"
                )
                self.metrics.state = CircuitState.CLOSED
                self.metrics.state_change_time = time.time()
                self.metrics.failure_count = 0
                self.metrics.consecutive_failures = 0
                self.metrics.consecutive_successes = 0

    async def _record_success(self, response_time: float) -> None:
        """Record successful call."""
        async with self._lock:
            self.metrics.success_count += 1
            self.metrics.total_calls += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.response_times.append(response_time)
            # Keep response_times bounded to prevent memory leak
            if len(self.metrics.response_times) > 100:
                self.metrics.response_times.pop(0)

            # Track slow calls
            if response_time >= self.config.slow_call_threshold:
                self.metrics.slow_calls += 1

            # State transitions based on success
            if self.metrics.state == CircuitState.HALF_OPEN:
                self.metrics.half_open_calls += 1
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    # Transition to CLOSED (inline to avoid deadlock)
                    logger.info(
                        f"Circuit breaker '{self.name}' closing after successful recovery"
                    )
                    self.metrics.state = CircuitState.CLOSED
                    self.metrics.state_change_time = time.time()
                    self.metrics.failure_count = 0
                    self.metrics.consecutive_failures = 0
                    self.metrics.consecutive_successes = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        async with self._lock:
            self.metrics.failure_count += 1
            self.metrics.total_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {error}",
                extra={
                    "consecutive_failures": self.metrics.consecutive_failures,
                    "error_rate": self.metrics.error_rate,
                },
            )

            # State transitions based on failure
            if self.metrics.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens circuit (inline to avoid deadlock)
                logger.warning(
                    f"Circuit breaker '{self.name}' opening due to failures "
                    f"(consecutive: {self.metrics.consecutive_failures}, "
                    f"error_rate: {self.metrics.error_rate:.2%})"
                )
                self.metrics.state = CircuitState.OPEN
                self.metrics.state_change_time = time.time()
            elif self.metrics.state == CircuitState.CLOSED:
                # Check if we should open circuit
                meets_threshold = (
                    self.metrics.consecutive_failures >= self.config.failure_threshold
                )
                meets_throughput = (
                    self.metrics.total_calls >= self.config.minimum_throughput
                )

                if meets_threshold and meets_throughput:
                    # Transition to OPEN (inline to avoid deadlock)
                    logger.warning(
                        f"Circuit breaker '{self.name}' opening due to failures "
                        f"(consecutive: {self.metrics.consecutive_failures}, "
                        f"error_rate: {self.metrics.error_rate:.2%})"
                    )
                    self.metrics.state = CircuitState.OPEN
                    self.metrics.state_change_time = time.time()

    @asynccontextmanager
    async def protect(self) -> AsyncIterator[None]:
        """Context manager for protected execution.

        Raises:
            CircuitBreakerError: If circuit is open and blocking requests

        Yields:
            None

        Example:
            async with circuit_breaker.protect():
                result = await risky_operation()
        """
        # Check current state and handle accordingly
        if self.metrics.state == CircuitState.OPEN:
            # Check if we should attempt reset
            if await self._should_attempt_reset():
                await self._transition_to_half_open()
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN, blocking requests"
                )

        # Increment in-flight counter for HALF_OPEN state (before yielding)
        is_half_open = self.metrics.state == CircuitState.HALF_OPEN
        if is_half_open:
            async with self._lock:
                # Limit concurrent calls in half-open state using in-flight counter
                if self._half_open_in_flight >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN, "
                        "max concurrent calls reached"
                    )
                self._half_open_in_flight += 1

        # Execute protected operation
        start_time = time.time()
        try:
            # Apply timeout to operation
            yield

            # Record success
            response_time = time.time() - start_time
            await self._record_success(response_time)

        except TimeoutError as e:
            # Timeout counts as failure
            await self._record_failure(e)
            raise

        except Exception as e:
            # Record failure
            await self._record_failure(e)
            raise

        finally:
            # Decrement in-flight counter for HALF_OPEN state
            if is_half_open:
                async with self._lock:
                    self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            timeout: Optional timeout override
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            asyncio.TimeoutError: If operation times out
            Exception: If func raises an exception
        """
        operation_timeout = timeout or self.config.timeout

        async with self.protect():
            try:
                result: T = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=operation_timeout,
                )
                return result
            except TimeoutError:
                logger.warning(
                    f"Circuit breaker '{self.name}' operation timed out "
                    f"after {operation_timeout}s"
                )
                raise

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics."""
        return self.metrics

    async def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        async with self._lock:
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")
            self.metrics = CircuitBreakerMetrics(name=self.name)
            self._half_open_in_flight = 0  # Reset in-flight counter

    async def force_open(self) -> None:
        """Manually force circuit breaker to OPEN state."""
        async with self._lock:
            logger.warning(f"Circuit breaker '{self.name}' manually forced to OPEN")
            self.metrics.state = CircuitState.OPEN
            self.metrics.state_change_time = time.time()
            self.metrics.last_failure_time = time.time()

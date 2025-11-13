"""
Database Circuit Breaker Implementation.

This module provides a circuit breaker pattern for database connectivity failures,
preventing cascading failures by monitoring database operation failures and
temporarily blocking requests when failure thresholds are exceeded.

Circuit Breaker States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing state, all requests are rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed

State Transitions:
    CLOSED → OPEN: After failure_threshold consecutive failures
    OPEN → HALF_OPEN: After timeout_seconds have elapsed
    HALF_OPEN → CLOSED: After 2 consecutive successful operations
    HALF_OPEN → OPEN: If any operation fails during testing

Performance:
    - State check: < 1ms
    - Metrics collection: < 0.5ms
    - Thread-safe with asyncio.Lock

Example Usage:
    ```python
    from circuit_breaker import DatabaseCircuitBreaker

    # Initialize circuit breaker
    circuit_breaker = DatabaseCircuitBreaker(
        failure_threshold=5,      # Open after 5 failures
        timeout_seconds=60,       # Wait 60s before retry
        half_open_max_calls=3,   # Allow 3 test calls
    )

    # Execute database operation with protection
    try:
        result = await circuit_breaker.execute(
            connection_manager.execute_query,
            "SELECT * FROM metadata_stamps WHERE file_hash = $1",
            file_hash,
            timeout=5.0,
        )
        print(f"Query result: {result}")
    except CircuitBreakerOpenError:
        print("Circuit breaker is OPEN - database temporarily unavailable")
    except Exception as e:
        print(f"Operation failed: {e}")

    # Check circuit breaker state
    state = circuit_breaker.get_state()
    print(f"Current state: {state}")

    # Get metrics
    metrics = circuit_breaker.get_metrics()
    print(f"Metrics: {metrics}")
    ```

Author: OmniNode Bridge Team
Created: October 7, 2025
Version: 1.0.0
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states for database connectivity failures."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is in OPEN state."""

    def __init__(self, message: str = "Circuit breaker is OPEN"):
        self.message = message
        super().__init__(self.message)


class DatabaseCircuitBreaker:
    """
    Circuit breaker implementation for database connectivity failures.

    Prevents cascading failures by monitoring database operation failures
    and temporarily blocking requests when failure thresholds are exceeded.

    State Machine:
        CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing) → CLOSED

    Attributes:
        state (CircuitBreakerState): Current circuit breaker state
        failure_count (int): Consecutive failure count in current state
        success_count (int): Consecutive success count (used in HALF_OPEN)
        last_failure_time (datetime | None): Timestamp of last failure
        half_open_calls (int): Number of calls made in HALF_OPEN state
        failure_threshold (int): Failures required to open circuit
        timeout_seconds (int): Seconds to wait before attempting HALF_OPEN
        half_open_max_calls (int): Max concurrent calls in HALF_OPEN state
        half_open_success_threshold (int): Successes needed to close circuit

    Performance Targets:
        - State check: < 1ms
        - Metrics collection: < 0.5ms
        - State transition: < 2ms

    Thread Safety:
        All state mutations are protected by asyncio.Lock for thread-safe operation.

    Example:
        ```python
        circuit_breaker = DatabaseCircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60,
            half_open_max_calls=3,
        )

        # Execute protected operation
        result = await circuit_breaker.execute(
            db_operation,
            *args,
            **kwargs,
        )
        ```
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3,
        half_open_success_threshold: int = 2,
    ):
        """
        Initialize the database circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures to open circuit
            timeout_seconds: Seconds to wait before attempting recovery
            half_open_max_calls: Maximum concurrent calls in HALF_OPEN state
            half_open_success_threshold: Consecutive successes to close circuit

        Raises:
            ValueError: If any threshold is <= 0
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be > 0")
        if half_open_success_threshold <= 0:
            raise ValueError("half_open_success_threshold must be > 0")

        # Configuration
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.half_open_success_threshold = half_open_success_threshold

        # State
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.half_open_calls = 0

        # Metrics
        self._total_failures = 0
        self._total_successes = 0
        self._state_transitions = 0
        self._last_state_change: datetime | None = None

        # Thread safety
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(
        cls, config: dict[str, Any] | None = None
    ) -> "DatabaseCircuitBreaker":
        """
        Create DatabaseCircuitBreaker from configuration dictionary.

        This factory method supports configuration from ModelContainer,
        allowing production tuning via environment variables or container config.

        Args:
            config: Configuration dictionary with circuit breaker settings.
                   Expected keys:
                   - failure_threshold (int): Failures to open circuit (default: 5)
                   - recovery_timeout (int): Seconds before retry (default: 60)
                   - half_open_max_calls (int): Max calls in half-open (default: 3)

        Returns:
            DatabaseCircuitBreaker instance configured from dictionary

        Example:
            ```python
            # From ModelContainer config
            container = ModelContainer()
            circuit_breaker_config = container.config.database.circuit_breaker()
            circuit_breaker = DatabaseCircuitBreaker.from_config(circuit_breaker_config)

            # Or with explicit config dict
            config = {
                "failure_threshold": 10,
                "recovery_timeout": 120,
                "half_open_max_calls": 5
            }
            circuit_breaker = DatabaseCircuitBreaker.from_config(config)
            ```
        """
        if config is None:
            config = {}

        return cls(
            failure_threshold=config.get("failure_threshold", 5),
            timeout_seconds=config.get("recovery_timeout", 60),
            half_open_max_calls=config.get("half_open_max_calls", 3),
            half_open_success_threshold=config.get("half_open_success_threshold", 2),
        )

    async def execute(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute an operation with circuit breaker protection.

        This method wraps the provided operation and applies circuit breaker
        logic to prevent cascading failures. If the circuit is OPEN, the
        operation is rejected immediately. Otherwise, the operation is
        executed and success/failure is recorded.

        State Transitions:
            - CLOSED: Execute operation, record result
            - OPEN: Reject immediately with CircuitBreakerOpenError
            - HALF_OPEN: Allow limited execution for testing

        Args:
            operation: Callable (async or sync) to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result from the operation if successful

        Raises:
            CircuitBreakerOpenError: If circuit breaker is in OPEN state
            Exception: Any exception raised by the operation

        Performance:
            - OPEN state rejection: < 1ms
            - CLOSED state execution: < 5ms overhead
            - HALF_OPEN state execution: < 10ms overhead

        Example:
            ```python
            # Execute database query with protection
            result = await circuit_breaker.execute(
                connection_manager.execute_query,
                "SELECT * FROM table WHERE id = $1",
                record_id,
                timeout=5.0,
            )
            ```
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. "
                        f"Will retry after {self.timeout_seconds}s timeout. "
                        f"Last failure: {self.last_failure_time.isoformat() if self.last_failure_time else 'unknown'}"
                    )

            # Check HALF_OPEN call limit
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is HALF_OPEN with max concurrent calls reached ({self.half_open_max_calls})"
                    )
                self.half_open_calls += 1

        # Execute the operation (outside lock to avoid blocking)
        try:
            # Handle both async and sync operations
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)

            # Record success
            await self.record_success()
            return result

        except Exception as e:
            # Record failure
            await self.record_failure()
            raise

        finally:
            # Decrement half_open_calls if in HALF_OPEN state
            if self.state == CircuitBreakerState.HALF_OPEN:
                async with self._lock:
                    self.half_open_calls = max(0, self.half_open_calls - 1)

    async def record_success(self) -> None:
        """
        Record a successful operation and update circuit breaker state.

        State Transitions:
            - CLOSED: Reset failure_count to 0
            - HALF_OPEN: Increment success_count, close if threshold reached
            - OPEN: No effect (should not be called in this state)

        Performance:
            - Execution time: < 1ms

        Example:
            ```python
            # Manually record success (usually automatic via execute())
            await circuit_breaker.record_success()
            ```
        """
        async with self._lock:
            self._total_successes += 1

            if self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Increment success count
                self.success_count += 1

                # Check if we should close the circuit
                if self.success_count >= self.half_open_success_threshold:
                    await self._transition_to_closed()

    async def record_failure(self) -> None:
        """
        Record a failed operation and update circuit breaker state.

        State Transitions:
            - CLOSED: Increment failure_count, open if threshold reached
            - HALF_OPEN: Immediately transition to OPEN
            - OPEN: Increment failure_count

        Performance:
            - Execution time: < 2ms

        Example:
            ```python
            # Manually record failure (usually automatic via execute())
            await circuit_breaker.record_failure()
            ```
        """
        async with self._lock:
            self._total_failures += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now(UTC)

            if self.state == CircuitBreakerState.CLOSED:
                # Check if we should open the circuit
                if self.failure_count >= self.failure_threshold:
                    await self._transition_to_open()

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Immediately open on any failure in HALF_OPEN
                await self._transition_to_open()

    def get_state(self) -> CircuitBreakerState:
        """
        Get the current circuit breaker state.

        Returns:
            Current CircuitBreakerState (CLOSED, OPEN, or HALF_OPEN)

        Performance:
            - Execution time: < 0.1ms

        Example:
            ```python
            state = circuit_breaker.get_state()
            if state == CircuitBreakerState.OPEN:
                print("Database is currently unavailable")
            ```
        """
        return self.state

    def get_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive circuit breaker metrics.

        Returns metrics including state, counts, timestamps, and configuration.

        Returns:
            Dictionary with circuit breaker metrics:
                - state: Current state (CLOSED, OPEN, HALF_OPEN)
                - failure_count: Consecutive failures in current state
                - success_count: Consecutive successes (HALF_OPEN only)
                - total_failures: Total failures since initialization
                - total_successes: Total successes since initialization
                - state_transitions: Number of state transitions
                - last_failure_time: ISO timestamp of last failure
                - last_state_change: ISO timestamp of last state change
                - half_open_calls: Current concurrent calls in HALF_OPEN
                - config: Configuration parameters

        Performance:
            - Execution time: < 0.5ms

        Example:
            ```python
            metrics = circuit_breaker.get_metrics()
            print(f"State: {metrics['state']}")
            print(f"Failure rate: {metrics['total_failures'] / (metrics['total_successes'] + metrics['total_failures'])}")
            print(f"State transitions: {metrics['state_transitions']}")
            ```
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "state_transitions": self._state_transitions,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": (
                self._last_state_change.isoformat() if self._last_state_change else None
            ),
            "half_open_calls": self.half_open_calls,
            "config": {
                "failure_threshold": self.failure_threshold,
                "timeout_seconds": self.timeout_seconds,
                "half_open_max_calls": self.half_open_max_calls,
                "half_open_success_threshold": self.half_open_success_threshold,
            },
        }

    def _should_attempt_reset(self) -> bool:
        """
        Check if enough time has passed to attempt circuit reset.

        Returns:
            True if timeout has elapsed since last failure

        Performance:
            - Execution time: < 0.1ms
        """
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

    async def _transition_to_open(self) -> None:
        """
        Transition circuit breaker to OPEN state.

        This is called when:
            - failure_threshold is reached in CLOSED state
            - Any failure occurs in HALF_OPEN state

        Performance:
            - Execution time: < 1ms
        """
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0
        self.half_open_calls = 0
        self._state_transitions += 1
        self._last_state_change = datetime.now(UTC)

    async def _transition_to_half_open(self) -> None:
        """
        Transition circuit breaker to HALF_OPEN state.

        This is called when:
            - timeout_seconds has elapsed in OPEN state

        Performance:
            - Execution time: < 1ms
        """
        self.state = CircuitBreakerState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self._state_transitions += 1
        self._last_state_change = datetime.now(UTC)

    async def _transition_to_closed(self) -> None:
        """
        Transition circuit breaker to CLOSED state.

        This is called when:
            - half_open_success_threshold is reached in HALF_OPEN state

        Performance:
            - Execution time: < 1ms
        """
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self._state_transitions += 1
        self._last_state_change = datetime.now(UTC)


# Example Usage and Testing
if __name__ == "__main__":

    async def mock_database_operation(should_fail: bool = False) -> str:
        """Mock database operation for testing."""
        await asyncio.sleep(0.01)  # Simulate DB latency
        if should_fail:
            raise Exception("Database connection failed")
        return "Query successful"

    async def demo_circuit_breaker():
        """Demonstrate circuit breaker functionality."""
        print("=" * 60)
        print("Database Circuit Breaker Demo")
        print("=" * 60)

        # Initialize circuit breaker
        circuit_breaker = DatabaseCircuitBreaker(
            failure_threshold=5,
            timeout_seconds=3,  # Shorter for demo
            half_open_max_calls=2,
        )

        # Test 1: Normal operation (CLOSED state)
        print("\n[Test 1] Normal Operation - CLOSED State")
        for i in range(3):
            try:
                result = await circuit_breaker.execute(
                    mock_database_operation, should_fail=False
                )
                print(f"  Operation {i+1}: {result}")
            except Exception as e:
                print(f"  Operation {i+1} failed: {e}")

        metrics = circuit_breaker.get_metrics()
        print(f"  State: {metrics['state']}, Successes: {metrics['total_successes']}")

        # Test 2: Trigger circuit breaker (OPEN state)
        print("\n[Test 2] Triggering Circuit Breaker - Opening Circuit")
        for i in range(6):
            try:
                result = await circuit_breaker.execute(
                    mock_database_operation, should_fail=True
                )
                print(f"  Operation {i+1}: {result}")
            except CircuitBreakerOpenError as e:
                print(f"  Operation {i+1}: Circuit OPEN - {e}")
            except Exception as e:
                print(f"  Operation {i+1} failed: {e}")

        metrics = circuit_breaker.get_metrics()
        print(f"  State: {metrics['state']}, Failures: {metrics['total_failures']}")

        # Test 3: Circuit stays open
        print("\n[Test 3] Circuit Stays Open - Rejecting Requests")
        for i in range(3):
            try:
                result = await circuit_breaker.execute(
                    mock_database_operation, should_fail=False
                )
                print(f"  Operation {i+1}: {result}")
            except CircuitBreakerOpenError as e:
                print(f"  Operation {i+1}: Rejected - Circuit OPEN")

        # Test 4: Wait for timeout and recovery (HALF_OPEN)
        print(f"\n[Test 4] Waiting {circuit_breaker.timeout_seconds}s for timeout...")
        await asyncio.sleep(circuit_breaker.timeout_seconds + 0.5)

        print("Testing recovery (HALF_OPEN state):")
        for i in range(3):
            try:
                result = await circuit_breaker.execute(
                    mock_database_operation, should_fail=False
                )
                print(f"  Operation {i+1}: {result}")
                metrics = circuit_breaker.get_metrics()
                print(
                    f"    State: {metrics['state']}, Success count: {metrics['success_count']}"
                )
            except CircuitBreakerOpenError as e:
                print(f"  Operation {i+1}: {e}")

        # Final metrics
        print("\n[Final Metrics]")
        final_metrics = circuit_breaker.get_metrics()
        print(f"  State: {final_metrics['state']}")
        print(f"  Total Successes: {final_metrics['total_successes']}")
        print(f"  Total Failures: {final_metrics['total_failures']}")
        print(f"  State Transitions: {final_metrics['state_transitions']}")
        print(f"  Last State Change: {final_metrics['last_state_change']}")

        print("\n" + "=" * 60)

    # Run the demo
    asyncio.run(demo_circuit_breaker())

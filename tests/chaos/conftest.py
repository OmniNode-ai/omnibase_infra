# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for chaos tests.

Provides chaos-specific fixtures for OMN-955 including:
- Chaos injection utilities (failure, timeout, partition simulation)
- Mock infrastructure clients with configurable failure modes
- Effect executors with chaos injection capability
- Event bus mocks with network partition simulation

Usage:
    Fixtures are automatically available to all tests in this package.
    Import additional models directly in test files as needed.

Example:
    >>> async def test_handler_failure(
    ...     chaos_effect_executor,
    ...     failure_injector,
    ... ):
    ...     # Configure 50% failure rate
    ...     failure_injector.set_failure_rate(0.5)
    ...     # Execute with chaos injection
    ...     result = await chaos_effect_executor.execute(...)

Related Tickets:
    - OMN-955: Chaos scenario tests
    - OMN-954: Effect idempotency
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)
from omnibase_infra.idempotency import InMemoryIdempotencyStore

# =============================================================================
# Chaos Injection Models
# =============================================================================


@dataclass
class ChaosConfig:
    """Configuration for chaos injection.

    Attributes:
        failure_rate: Probability of failure (0.0-1.0).
        timeout_rate: Probability of timeout (0.0-1.0).
        latency_min_ms: Minimum latency injection in milliseconds.
        latency_max_ms: Maximum latency injection in milliseconds.
        partition_duration_ms: Duration of simulated partition in milliseconds.
    """

    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    latency_min_ms: int = 0
    latency_max_ms: int = 0
    partition_duration_ms: int = 0
    enabled: bool = True


@dataclass
class FailureInjector:
    """Utility for injecting failures into operations.

    This class provides methods for injecting various failure modes into
    operations, simulating real-world failure scenarios.

    Attributes:
        config: Chaos configuration.
        failure_count: Number of failures injected.
        timeout_count: Number of timeouts injected.
    """

    config: ChaosConfig = field(default_factory=ChaosConfig)
    failure_count: int = 0
    timeout_count: int = 0

    def set_failure_rate(self, rate: float) -> None:
        """Set the failure injection rate.

        Args:
            rate: Probability of failure (0.0-1.0).
        """
        self.config.failure_rate = max(0.0, min(1.0, rate))

    def set_timeout_rate(self, rate: float) -> None:
        """Set the timeout injection rate.

        Args:
            rate: Probability of timeout (0.0-1.0).
        """
        self.config.timeout_rate = max(0.0, min(1.0, rate))

    def set_latency_range(self, min_ms: int, max_ms: int) -> None:
        """Set the latency injection range.

        Args:
            min_ms: Minimum latency in milliseconds.
            max_ms: Maximum latency in milliseconds.
        """
        self.config.latency_min_ms = min_ms
        self.config.latency_max_ms = max_ms

    def should_fail(self) -> bool:
        """Determine if the current operation should fail.

        Returns:
            True if operation should fail based on failure_rate.
        """
        if not self.config.enabled:
            return False
        return random.random() < self.config.failure_rate

    def should_timeout(self) -> bool:
        """Determine if the current operation should timeout.

        Returns:
            True if operation should timeout based on timeout_rate.
        """
        if not self.config.enabled:
            return False
        return random.random() < self.config.timeout_rate

    async def maybe_inject_failure(
        self,
        operation: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Possibly inject a failure into the operation.

        Args:
            operation: Name of the operation being executed.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            ValueError: If failure injection triggers.
        """
        if self.should_fail():
            self.failure_count += 1
            raise ValueError(
                f"Chaos injection: simulated failure in '{operation}' "
                f"(correlation_id={correlation_id})"
            )

    async def maybe_inject_timeout(
        self,
        operation: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Possibly inject a timeout into the operation.

        Args:
            operation: Name of the operation being executed.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            InfraTimeoutError: If timeout injection triggers.
        """
        if self.should_timeout():
            self.timeout_count += 1
            context = ModelInfraErrorContext(
                operation=operation,
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Chaos injection: simulated timeout in '{operation}'",
                context=context,
            )

    async def maybe_inject_latency(self) -> None:
        """Possibly inject latency into the operation."""
        if not self.config.enabled:
            return
        if self.config.latency_min_ms > 0 or self.config.latency_max_ms > 0:
            latency_ms = random.randint(
                self.config.latency_min_ms,
                self.config.latency_max_ms,
            )
            await asyncio.sleep(latency_ms / 1000.0)

    def reset_counts(self) -> None:
        """Reset failure and timeout counters."""
        self.failure_count = 0
        self.timeout_count = 0


@dataclass
class NetworkPartitionSimulator:
    """Simulator for network partitions in event bus testing.

    Manage simulated network partition state for testing how the system
    handles connectivity issues.

    Thread Safety:
        This simulator is designed for **single asyncio event loop** usage with
        cooperative concurrency. It is NOT thread-safe for multi-threaded access.

        The ``asyncio.Lock`` (``self._lock``) protects state mutations in
        ``simulate_partition_healing`` to prevent race conditions when multiple
        concurrent calls attempt to modify ``is_partitioned``.

        Lock Scope:
            - PROTECTED: ``end_partition()`` call (modifies is_partitioned state)
            - NOT PROTECTED: Reconnection callbacks (may trigger I/O operations)

        Callbacks execute outside the lock to avoid potential deadlocks if callbacks
        attempt to acquire the same lock or perform long-running I/O.

    Attributes:
        is_partitioned: Whether a partition is currently active.
        partition_start_time: When the current partition started.
        reconnection_callbacks: Callbacks to invoke on reconnection.
    """

    is_partitioned: bool = False
    partition_start_time: float | None = None
    reconnection_callbacks: list[Callable[[], Awaitable[None]]] = field(
        default_factory=list
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def start_partition(self) -> None:
        """Start a network partition simulation."""
        self.is_partitioned = True
        self.partition_start_time = asyncio.get_event_loop().time()

    def end_partition(self) -> None:
        """End the network partition simulation."""
        self.is_partitioned = False
        self.partition_start_time = None

    async def simulate_partition_healing(
        self,
        duration_ms: int = 100,
    ) -> None:
        """Simulate partition healing with delay.

        Thread Safety:
            Acquires ``_lock`` before calling ``end_partition()`` to ensure
            atomic state transition. Callbacks execute outside the lock to
            allow concurrent I/O operations without holding the lock.

        Args:
            duration_ms: Duration to wait before healing partition.
        """
        await asyncio.sleep(duration_ms / 1000.0)

        # Acquire lock for atomic state transition
        async with self._lock:
            self.end_partition()

        # Invoke reconnection callbacks outside lock (may trigger I/O)
        for callback in self.reconnection_callbacks:
            await callback()

    def add_reconnection_callback(
        self, callback: Callable[[], Awaitable[None]]
    ) -> None:
        """Add a callback to invoke on reconnection.

        Args:
            callback: Async callback to invoke (must be callable returning Awaitable).
        """
        self.reconnection_callbacks.append(callback)


# =============================================================================
# Chaos Effect Executor
# =============================================================================


class ChaosEffectExecutor:
    """Effect executor with chaos injection capability.

    This class wraps effect execution with configurable chaos injection,
    allowing tests to simulate various failure scenarios.

    Thread Safety:
        This executor is designed for **single asyncio event loop** usage with
        cooperative concurrency. It is NOT thread-safe for multi-threaded access.

        The ``asyncio.Lock`` (``self._lock``) provides atomicity for counter updates
        ONLY. The lock is NOT held during the entire ``execute_with_chaos`` operation,
        meaning concurrent coroutines can interleave their execution phases.

        Lock Scope:
            - PROTECTED: ``execution_count`` and ``failed_count`` increments
            - NOT PROTECTED: Idempotency checks, chaos injection, backend execution

    Concurrency Considerations:
        Multiple concurrent calls to ``execute_with_chaos`` are supported within
        a single asyncio event loop. Each call will:

        1. Independently check idempotency (no lock held)
        2. Execute backend operations concurrently (no lock held)
        3. Atomically update counters (lock acquired/released per update)

        This design prioritizes throughput over strict serialization. If you need
        strictly ordered execution, serialize calls externally.

    Counter Accuracy:
        Counters are **eventually accurate** - they correctly reflect the total
        number of successes and failures, but assertions on counter values during
        active concurrent execution may observe intermediate states.

        Warning:
            In concurrent test scenarios, avoid asserting on counters until ALL
            concurrent operations have completed. Use ``asyncio.gather()`` to
            await all operations before checking counters.

    Test Isolation Recommendation:
        For deterministic testing, prefer one of these patterns:

        1. **Sequential execution**: Await each ``execute_with_chaos`` individually
        2. **Gather then assert**: Use ``asyncio.gather()`` for concurrent ops,
           then assert on counters after gather completes
        3. **Fresh executor per test**: Use the ``chaos_effect_executor`` fixture
           which provides a fresh instance per test

    Example - Safe Concurrent Testing::

        # CORRECT: Wait for all operations before asserting
        results = await asyncio.gather(
            executor.execute_with_chaos(uuid4(), "op1", ...),
            executor.execute_with_chaos(uuid4(), "op2", ...),
            executor.execute_with_chaos(uuid4(), "op3", ...),
            return_exceptions=True,
        )
        # Now counters are stable
        assert executor.execution_count + executor.failed_count == 3

        # INCORRECT: Asserting during active execution
        task = asyncio.create_task(executor.execute_with_chaos(...))
        assert executor.execution_count == 0  # May be 0 or 1 - race condition!
        await task

    Attributes:
        idempotency_store: Store for idempotency checking.
        failure_injector: Injector for failure simulation.
        backend_client: Mock backend client for recording calls.
        execution_count: Number of successful executions.
        failed_count: Number of failed executions.
    """

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        failure_injector: FailureInjector,
        backend_client: MagicMock,
    ) -> None:
        """Initialize the chaos effect executor.

        Args:
            idempotency_store: Store for idempotency checking.
            failure_injector: Injector for failure simulation.
            backend_client: Mock backend client.
        """
        self.idempotency_store = idempotency_store
        self.failure_injector = failure_injector
        self.backend_client = backend_client
        self.execution_count = 0
        self.failed_count = 0
        # Lock for atomic counter updates only. NOT held during execution.
        # See class docstring "Thread Safety" section for details.
        self._lock = asyncio.Lock()

    async def execute_with_chaos(
        self,
        intent_id: UUID,
        operation: str,
        domain: str = "chaos",
        correlation_id: UUID | None = None,
        fail_point: str | None = None,
    ) -> bool:
        """Execute an operation with chaos injection.

        Args:
            intent_id: Unique identifier for this intent.
            operation: Name of the operation.
            domain: Idempotency domain.
            correlation_id: Optional correlation ID.
            fail_point: Specific point to inject failure ("pre", "mid", "post").

        Returns:
            True if operation succeeded (or was idempotent duplicate).

        Raises:
            ValueError: If chaos injection triggers failure.
            InfraTimeoutError: If chaos injection triggers timeout.
        """
        # --- PHASE 1: Pre-execution chaos injection (NO LOCK HELD) ---
        # Multiple concurrent calls can reach this point simultaneously.
        # Chaos injection is intentionally non-atomic to simulate real failures.
        if fail_point == "pre":
            await self.failure_injector.maybe_inject_failure(
                f"{operation}:pre",
                correlation_id,
            )
            await self.failure_injector.maybe_inject_timeout(
                f"{operation}:pre",
                correlation_id,
            )

        # --- PHASE 2: Idempotency check (NO LOCK HELD) ---
        # The idempotency store has its own internal synchronization.
        # Concurrent calls with different intent_ids will proceed independently.
        # Concurrent calls with the SAME intent_id rely on store atomicity.
        is_new = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain=domain,
            correlation_id=correlation_id,
        )

        if not is_new:
            # Duplicate detected - skip execution, no counter update needed.
            # This is an early return; counters only track actual execution attempts.
            return True

        # --- PHASE 3: Execution with chaos (NO LOCK HELD) ---
        # Main execution phase runs without lock to allow concurrent operations.
        # Lock is only acquired briefly for counter updates.
        try:
            # Mid-execution chaos injection (inside try block to track failures)
            if fail_point == "mid":
                await self.failure_injector.maybe_inject_failure(
                    f"{operation}:mid",
                    correlation_id,
                )
                await self.failure_injector.maybe_inject_timeout(
                    f"{operation}:mid",
                    correlation_id,
                )

            # Inject latency if configured (simulates slow operations)
            await self.failure_injector.maybe_inject_latency()

            # Execute backend operation (actual effect - may take significant time)
            await self.backend_client.execute(operation, intent_id)

            # --- COUNTER UPDATE: Lock acquired briefly for atomic increment ---
            # Lock scope: ONLY the counter increment, NOT the surrounding code.
            # Lock is released immediately after increment completes.
            async with self._lock:
                self.execution_count += 1
            # --- Lock released here ---

            # Post-execution chaos injection (after success counter updated)
            if fail_point == "post":
                await self.failure_injector.maybe_inject_failure(
                    f"{operation}:post",
                    correlation_id,
                )
                await self.failure_injector.maybe_inject_timeout(
                    f"{operation}:post",
                    correlation_id,
                )

            return True

        except Exception:
            # --- FAILURE COUNTER UPDATE: Lock acquired briefly for atomic increment ---
            # Even on failure, we atomically update the failure counter.
            # Note: If post-execution chaos triggers AFTER success counter was
            # incremented, both counters may be updated for the same operation.
            async with self._lock:
                self.failed_count += 1
            # --- Lock released here ---
            raise

    def reset_counts(self) -> None:
        """Reset execution counters.

        Warning:
            This method is NOT thread-safe. Only call when no concurrent
            ``execute_with_chaos`` operations are in progress. Typically
            called at test setup/teardown boundaries.
        """
        # No lock acquired - caller must ensure no concurrent operations.
        # Safe to call between test cases when executor is idle.
        self.execution_count = 0
        self.failed_count = 0


# =============================================================================
# Mock Event Bus with Partition Simulation
# =============================================================================


class MockEventBusWithPartition:
    """Mock event bus that can simulate network partitions.

    This class provides a mock event bus implementation that can simulate
    network partitions and reconnection behavior for testing.

    Attributes:
        partition_simulator: Simulator for partition state.
        published_messages: List of published messages.
        subscribers: Dict of topic -> handlers.
        started: Whether the bus is started.
    """

    def __init__(self, partition_simulator: NetworkPartitionSimulator) -> None:
        """Initialize the mock event bus.

        Args:
            partition_simulator: Simulator for partition state.
        """
        self.partition_simulator = partition_simulator
        self.published_messages: list[dict[str, object]] = []
        self.subscribers: dict[str, list[Callable]] = {}
        self.started = False
        self.connection_attempts = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the event bus."""
        if self.partition_simulator.is_partitioned:
            self.connection_attempts += 1
            raise InfraConnectionError(
                "Chaos injection: network partition active",
                context=ModelInfraErrorContext(operation="start"),
            )
        self.started = True

    async def close(self) -> None:
        """Close the event bus."""
        self.started = False

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        """Publish a message to a topic.

        Args:
            topic: Topic to publish to.
            key: Optional message key.
            value: Message value.

        Raises:
            InfraUnavailableError: If bus not started.
            InfraConnectionError: If partition is active.
        """
        if not self.started:
            raise InfraUnavailableError(
                "Event bus not started",
                context=ModelInfraErrorContext(operation="publish"),
            )

        if self.partition_simulator.is_partitioned:
            raise InfraConnectionError(
                "Chaos injection: network partition during publish",
                context=ModelInfraErrorContext(operation="publish"),
            )

        async with self._lock:
            self.published_messages.append(
                {
                    "topic": topic,
                    "key": key,
                    "value": value,
                }
            )

        # Notify subscribers
        if topic in self.subscribers:
            for handler in self.subscribers[topic]:
                await handler({"topic": topic, "key": key, "value": value})

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: Callable,
    ) -> Callable:
        """Subscribe to a topic.

        Args:
            topic: Topic to subscribe to.
            group: Consumer group.
            handler: Handler callback.

        Returns:
            Unsubscribe function.
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)

        async def unsubscribe() -> None:
            if topic in self.subscribers and handler in self.subscribers[topic]:
                self.subscribers[topic].remove(handler)

        return unsubscribe

    async def health_check(self) -> dict[str, object]:
        """Check event bus health.

        Returns:
            Health status dict.
        """
        return {
            "healthy": self.started and not self.partition_simulator.is_partitioned,
            "started": self.started,
            "partitioned": self.partition_simulator.is_partitioned,
            "message_count": len(self.published_messages),
        }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def chaos_config() -> ChaosConfig:
    """Create default chaos configuration.

    Returns:
        ChaosConfig with default settings (no chaos by default).
    """
    return ChaosConfig()


@pytest.fixture
def failure_injector(chaos_config: ChaosConfig) -> FailureInjector:
    """Create failure injector with chaos configuration.

    Args:
        chaos_config: Chaos configuration fixture.

    Returns:
        FailureInjector configured for chaos testing.
    """
    return FailureInjector(config=chaos_config)


@pytest.fixture
def high_failure_injector() -> FailureInjector:
    """Create failure injector with high failure rate.

    Returns:
        FailureInjector configured with 50% failure rate.
    """
    config = ChaosConfig(failure_rate=0.5)
    return FailureInjector(config=config)


@pytest.fixture
def deterministic_failure_injector() -> FailureInjector:
    """Create failure injector with 100% failure rate for deterministic tests.

    Returns:
        FailureInjector that always fails.
    """
    config = ChaosConfig(failure_rate=1.0)
    return FailureInjector(config=config)


@pytest.fixture
def network_partition_simulator() -> NetworkPartitionSimulator:
    """Create network partition simulator.

    Returns:
        NetworkPartitionSimulator for network chaos testing.
    """
    return NetworkPartitionSimulator()


@pytest.fixture
def mock_backend_client() -> MagicMock:
    """Create mock backend client for effect execution.

    Returns:
        MagicMock configured for async operations.
    """
    client = MagicMock()
    client.execute = AsyncMock(return_value=None)
    return client


@pytest.fixture
def chaos_idempotency_store() -> InMemoryIdempotencyStore:
    """Create in-memory idempotency store for chaos testing.

    Returns:
        InMemoryIdempotencyStore for testing.
    """
    return InMemoryIdempotencyStore()


@pytest.fixture
def chaos_effect_executor(
    chaos_idempotency_store: InMemoryIdempotencyStore,
    failure_injector: FailureInjector,
    mock_backend_client: MagicMock,
) -> ChaosEffectExecutor:
    """Create chaos effect executor.

    Args:
        chaos_idempotency_store: Idempotency store fixture.
        failure_injector: Failure injector fixture.
        mock_backend_client: Mock backend client fixture.

    Returns:
        ChaosEffectExecutor for testing.
    """
    return ChaosEffectExecutor(
        idempotency_store=chaos_idempotency_store,
        failure_injector=failure_injector,
        backend_client=mock_backend_client,
    )


@pytest.fixture
def mock_event_bus_with_partition(
    network_partition_simulator: NetworkPartitionSimulator,
) -> MockEventBusWithPartition:
    """Create mock event bus with partition simulation.

    Args:
        network_partition_simulator: Partition simulator fixture.

    Returns:
        MockEventBusWithPartition for network chaos testing.
    """
    return MockEventBusWithPartition(network_partition_simulator)


@pytest.fixture
async def started_event_bus_with_partition(
    mock_event_bus_with_partition: MockEventBusWithPartition,
) -> AsyncIterator[MockEventBusWithPartition]:
    """Create and start mock event bus with partition simulation.

    Args:
        mock_event_bus_with_partition: Mock event bus fixture.

    Yields:
        Started MockEventBusWithPartition.
    """
    await mock_event_bus_with_partition.start()
    yield mock_event_bus_with_partition
    await mock_event_bus_with_partition.close()


@pytest.fixture
def correlation_id() -> UUID:
    """Create a UUID correlation ID for request tracing.

    Returns:
        UUID: A fresh UUID4 for correlation tracking.
    """
    return uuid4()


# =============================================================================
# Common Test Utilities
# =============================================================================


async def gather_with_error_collection(
    coroutines: list,
    *,
    return_exceptions: bool = True,
) -> tuple[list[object], list[Exception]]:
    """Execute coroutines concurrently and separate successes from failures.

    This utility simplifies the common pattern of running multiple async
    operations and classifying their outcomes.

    Args:
        coroutines: List of coroutines to execute concurrently.
        return_exceptions: If True, exceptions are captured instead of raised.

    Returns:
        Tuple of (successful_results, exceptions).

    Example:
        >>> async def may_fail(i: int) -> str:
        ...     if i % 2 == 0:
        ...         raise ValueError("Even number")
        ...     return f"success_{i}"
        ...
        >>> successes, failures = await gather_with_error_collection(
        ...     [may_fail(i) for i in range(5)]
        ... )
        >>> assert len(successes) == 2  # 1, 3
        >>> assert len(failures) == 3   # 0, 2, 4
    """
    results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)

    successes: list[object] = []
    failures: list[Exception] = []

    for result in results:
        if isinstance(result, Exception):
            failures.append(result)
        else:
            successes.append(result)

    return successes, failures


def classify_results_by_type(
    results: list[object | Exception],
    *,
    success_type: type = bool,
) -> dict[str, list]:
    """Classify mixed results by their type.

    Useful for analyzing results from asyncio.gather(return_exceptions=True).

    Args:
        results: Mixed list of results and exceptions.
        success_type: Type to consider as success (default: bool for True values).

    Returns:
        Dict with keys: 'successes', 'failures', and exception class names.

    Example:
        >>> results = [True, ValueError("a"), True, InfraTimeoutError("b")]
        >>> classified = classify_results_by_type(results)
        >>> classified['successes']  # [True, True]
        >>> classified['ValueError']  # [ValueError("a")]
        >>> classified['InfraTimeoutError']  # [InfraTimeoutError("b")]
    """
    classified: dict[str, list] = {"successes": [], "failures": []}

    for result in results:
        if isinstance(result, Exception):
            classified["failures"].append(result)
            # Also categorize by exception type
            exc_type_name = type(result).__name__
            if exc_type_name not in classified:
                classified[exc_type_name] = []
            classified[exc_type_name].append(result)
        elif (success_type is bool and result is True) or isinstance(
            result, success_type
        ):
            classified["successes"].append(result)

    return classified


def assert_failure_rate_within_tolerance(
    actual_failures: int,
    total_attempts: int,
    expected_rate: float,
    tolerance: float = 0.2,
    *,
    context: str = "",
) -> None:
    """Assert that observed failure rate is within expected tolerance.

    Useful for chaos tests with probabilistic failure injection.

    Args:
        actual_failures: Number of observed failures.
        total_attempts: Total number of attempts.
        expected_rate: Expected failure rate (0.0-1.0).
        tolerance: Acceptable deviation from expected (default 0.2 = 20%).
        context: Optional context string for error message.

    Raises:
        AssertionError: If failure rate is outside tolerance.

    Example:
        >>> # With 30% failure rate and 100 attempts, expect ~30 failures
        >>> # With 20% tolerance, acceptable range is 24-36
        >>> assert_failure_rate_within_tolerance(
        ...     actual_failures=28,
        ...     total_attempts=100,
        ...     expected_rate=0.3,
        ...     tolerance=0.2,
        ...     context="random failure test",
        ... )
    """
    expected_failures = total_attempts * expected_rate
    min_failures = int(expected_failures * (1 - tolerance))
    max_failures = int(expected_failures * (1 + tolerance))
    actual_rate = actual_failures / total_attempts if total_attempts > 0 else 0.0

    context_prefix = f"{context}: " if context else ""

    assert min_failures <= actual_failures <= max_failures, (
        f"{context_prefix}Failure rate {actual_rate:.1%} ({actual_failures}/{total_attempts}) "
        f"outside expected range [{min_failures}, {max_failures}] "
        f"(target: {expected_rate:.0%} +/- {tolerance:.0%})"
    )


async def run_concurrent_with_tracking(
    operation: Callable[[int], Awaitable[object]],
    count: int,
    *,
    collect_exceptions: bool = True,
) -> tuple[list[object], list[Exception]]:
    """Run an async operation multiple times concurrently with result tracking.

    Simplifies the common pattern of running concurrent operations with
    thread-safe result collection.

    Args:
        operation: Async callable taking an index and returning a result.
        count: Number of times to execute the operation.
        collect_exceptions: If True, collect exceptions instead of raising.

    Returns:
        Tuple of (successful_results, exceptions).

    Example:
        >>> async def test_op(i: int) -> str:
        ...     if i % 3 == 0:
        ...         raise ValueError(f"Failed at {i}")
        ...     return f"success_{i}"
        ...
        >>> successes, failures = await run_concurrent_with_tracking(
        ...     test_op, count=10
        ... )
        >>> assert len(successes) + len(failures) == 10
    """
    results: list[object] = []
    errors: list[Exception] = []
    lock = asyncio.Lock()

    async def execute_one(i: int) -> None:
        try:
            result = await operation(i)
            async with lock:
                results.append(result)
        except Exception as e:
            if collect_exceptions:
                async with lock:
                    errors.append(e)
            else:
                raise

    await asyncio.gather(*[execute_one(i) for i in range(count)])
    return results, errors


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers for chaos tests.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "chaos: mark test as a chaos engineering test",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deferred for performance)",
    )

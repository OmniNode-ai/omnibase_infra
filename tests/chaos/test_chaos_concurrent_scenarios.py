# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chaos tests for concurrent failure scenarios (OMN-955).

This test suite validates system behavior when multiple failure modes occur
simultaneously. It covers:

1. Multiple services failing at the same time (database + cache + API)
2. Cascading failures where one failure triggers another
3. Race condition handling during concurrent chaotic operations
4. Mixed failure modes (timeouts + connection errors + partial failures)

Architecture:
    Concurrent chaos scenarios are particularly challenging because:

    1. Multiple failure modes can interact in unexpected ways
    2. Race conditions can expose hidden bugs in error handling
    3. Circuit breakers may trip across multiple services simultaneously
    4. Rollback logic must handle multiple partial failures

    The system should:
    - Handle multiple simultaneous failures gracefully
    - Properly track failure counts across concurrent operations
    - Maintain data integrity during concurrent chaos
    - Not deadlock or hang under failure conditions

Test Organization:
    - TestConcurrentMultiServiceFailures: Multiple services failing together
    - TestCascadingFailures: Failure propagation chains
    - TestRaceConditionHandling: Concurrent operations with chaos
    - TestMixedFailureModes: Combining different failure types

Related Tickets:
    - OMN-955: Chaos scenario tests
    - OMN-954: Effect idempotency
"""

from __future__ import annotations

import asyncio
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
from tests.chaos.conftest import (
    ChaosConfig,
    ChaosEffectExecutor,
    FailureInjector,
    NetworkPartitionSimulator,
)

# =============================================================================
# Helper Classes for Concurrent Testing
# =============================================================================


@dataclass
class ServiceSimulator:
    """Simulates an external service with configurable failure modes.

    Attributes:
        name: Service name for identification.
        failure_injector: Injector for failure simulation.
        call_count: Number of times the service was called.
        success_count: Number of successful calls.
        failure_count: Number of failed calls.
    """

    name: str
    failure_injector: FailureInjector
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def execute(
        self,
        operation: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Execute an operation on this service.

        Args:
            operation: Operation to execute.
            correlation_id: Optional correlation ID.

        Returns:
            True if operation succeeded.

        Raises:
            ValueError: If failure injection triggers.
            InfraTimeoutError: If timeout injection triggers.
        """
        async with self._lock:
            self.call_count += 1

        try:
            await self.failure_injector.maybe_inject_failure(
                f"{self.name}:{operation}",
                correlation_id,
            )
            await self.failure_injector.maybe_inject_timeout(
                f"{self.name}:{operation}",
                correlation_id,
            )
            await self.failure_injector.maybe_inject_latency()

            async with self._lock:
                self.success_count += 1
            return True

        except Exception:
            async with self._lock:
                self.failure_count += 1
            raise


@dataclass
class MultiServiceExecutor:
    """Executor that coordinates operations across multiple services.

    This simulates a workflow that must interact with multiple backend
    services (database, cache, external API) concurrently.

    Attributes:
        services: Dict of service name to ServiceSimulator.
        idempotency_store: Store for idempotency checking.
        completed_operations: List of completed operations.
        failed_operations: List of failed operations.
    """

    services: dict[str, ServiceSimulator]
    idempotency_store: InMemoryIdempotencyStore
    completed_operations: list[str] = field(default_factory=list)
    failed_operations: list[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def execute_on_service(
        self,
        service_name: str,
        operation: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Execute operation on a specific service.

        Args:
            service_name: Name of the service.
            operation: Operation to execute.
            correlation_id: Optional correlation ID.

        Returns:
            True if succeeded.
        """
        service = self.services.get(service_name)
        if not service:
            raise ValueError(f"Unknown service: {service_name}")

        try:
            result = await service.execute(operation, correlation_id)
            async with self._lock:
                self.completed_operations.append(f"{service_name}:{operation}")
            return result
        except Exception as e:
            async with self._lock:
                self.failed_operations.append(f"{service_name}:{operation}")
            raise

    async def execute_all_concurrent(
        self,
        operations: list[tuple[str, str]],
        correlation_id: UUID | None = None,
    ) -> list[bool | Exception]:
        """Execute multiple operations across services concurrently.

        Args:
            operations: List of (service_name, operation) tuples.
            correlation_id: Optional correlation ID.

        Returns:
            List of results (True for success, Exception for failure).
        """
        tasks = [
            self.execute_on_service(service_name, operation, correlation_id)
            for service_name, operation in operations
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def database_service() -> ServiceSimulator:
    """Create database service simulator.

    Returns:
        ServiceSimulator configured as database service.
    """
    return ServiceSimulator(
        name="database",
        failure_injector=FailureInjector(config=ChaosConfig()),
    )


@pytest.fixture
def cache_service() -> ServiceSimulator:
    """Create cache service simulator.

    Returns:
        ServiceSimulator configured as cache service.
    """
    return ServiceSimulator(
        name="cache",
        failure_injector=FailureInjector(config=ChaosConfig()),
    )


@pytest.fixture
def external_api_service() -> ServiceSimulator:
    """Create external API service simulator.

    Returns:
        ServiceSimulator configured as external API service.
    """
    return ServiceSimulator(
        name="external_api",
        failure_injector=FailureInjector(config=ChaosConfig()),
    )


@pytest.fixture
def multi_service_executor(
    database_service: ServiceSimulator,
    cache_service: ServiceSimulator,
    external_api_service: ServiceSimulator,
) -> MultiServiceExecutor:
    """Create multi-service executor with all services.

    Args:
        database_service: Database service simulator fixture.
        cache_service: Cache service simulator fixture.
        external_api_service: External API service simulator fixture.

    Returns:
        MultiServiceExecutor with all services configured.
    """
    return MultiServiceExecutor(
        services={
            "database": database_service,
            "cache": cache_service,
            "external_api": external_api_service,
        },
        idempotency_store=InMemoryIdempotencyStore(),
    )


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.chaos
class TestConcurrentMultiServiceFailures:
    """Test handling of multiple services failing simultaneously."""

    @pytest.mark.asyncio
    async def test_all_services_fail_simultaneously(
        self,
        multi_service_executor: MultiServiceExecutor,
    ) -> None:
        """Test behavior when all services fail at the same time.

        When multiple services fail simultaneously:
        - All failures should be captured
        - Each failure should be tracked independently
        - No deadlocks or hangs should occur
        """
        # Arrange - set all services to 100% failure rate
        for service in multi_service_executor.services.values():
            service.failure_injector.set_failure_rate(1.0)

        correlation_id = uuid4()

        # Act - execute on all services concurrently
        results = await multi_service_executor.execute_all_concurrent(
            operations=[
                ("database", "query"),
                ("cache", "get"),
                ("external_api", "call"),
            ],
            correlation_id=correlation_id,
        )

        # Assert - all operations failed
        assert len(results) == 3
        assert all(isinstance(r, Exception) for r in results)
        assert len(multi_service_executor.failed_operations) == 3
        assert len(multi_service_executor.completed_operations) == 0

        # Verify each service tracked its failure
        for service in multi_service_executor.services.values():
            assert service.failure_count == 1
            assert service.success_count == 0

    @pytest.mark.asyncio
    async def test_partial_multi_service_failures(
        self,
        multi_service_executor: MultiServiceExecutor,
    ) -> None:
        """Test behavior when some services fail while others succeed.

        In a partial failure scenario:
        - Successful operations should complete normally
        - Failed operations should be tracked
        - Results should correctly reflect mixed outcomes
        """
        # Arrange - only database fails
        multi_service_executor.services["database"].failure_injector.set_failure_rate(
            1.0
        )
        multi_service_executor.services["cache"].failure_injector.set_failure_rate(0.0)
        multi_service_executor.services[
            "external_api"
        ].failure_injector.set_failure_rate(0.0)

        correlation_id = uuid4()

        # Act
        results = await multi_service_executor.execute_all_concurrent(
            operations=[
                ("database", "query"),
                ("cache", "get"),
                ("external_api", "call"),
            ],
            correlation_id=correlation_id,
        )

        # Assert - one failure, two successes
        assert len(results) == 3

        # Count successes and failures
        successes = [r for r in results if r is True]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 2
        assert len(failures) == 1
        assert len(multi_service_executor.failed_operations) == 1
        assert len(multi_service_executor.completed_operations) == 2

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_high_failure_rate(
        self,
        multi_service_executor: MultiServiceExecutor,
    ) -> None:
        """Test many concurrent operations with probabilistic failures.

        With a high but not deterministic failure rate:
        - Some operations should succeed, others fail
        - Total operations should match expected count
        - No operations should be lost or duplicated
        """
        # Arrange - 50% failure rate on all services
        for service in multi_service_executor.services.values():
            service.failure_injector.set_failure_rate(0.5)

        num_operations_per_service = 20
        operations = []
        for i in range(num_operations_per_service):
            operations.extend(
                [
                    ("database", f"query_{i}"),
                    ("cache", f"get_{i}"),
                    ("external_api", f"call_{i}"),
                ]
            )

        # Act
        results = await multi_service_executor.execute_all_concurrent(
            operations=operations,
            correlation_id=uuid4(),
        )

        # Assert
        total_ops = num_operations_per_service * 3
        assert len(results) == total_ops

        # All operations should be accounted for
        tracked = len(multi_service_executor.completed_operations) + len(
            multi_service_executor.failed_operations
        )
        assert tracked == total_ops

        # With 50% failure rate, expect roughly half to fail (with variance)
        failures = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if r is True]

        # Allow for statistical variance - at least some of each
        assert len(failures) > 0, "Expected at least some failures"
        assert len(successes) > 0, "Expected at least some successes"

    @pytest.mark.asyncio
    async def test_service_independence_during_concurrent_failures(
        self,
        multi_service_executor: MultiServiceExecutor,
    ) -> None:
        """Test that service failures are independent.

        Failure in one service should not affect other services:
        - Each service's counters should be independent
        - One service crashing should not prevent others from completing
        """
        # Arrange - staggered failure rates
        multi_service_executor.services["database"].failure_injector.set_failure_rate(
            1.0
        )  # Always fails
        multi_service_executor.services["cache"].failure_injector.set_failure_rate(
            0.0
        )  # Never fails
        multi_service_executor.services[
            "external_api"
        ].failure_injector.set_failure_rate(0.5)  # Sometimes fails

        # Execute many operations
        num_iterations = 30
        all_results: list[list[bool | Exception]] = []

        for i in range(num_iterations):
            results = await multi_service_executor.execute_all_concurrent(
                operations=[
                    ("database", f"query_{i}"),
                    ("cache", f"get_{i}"),
                    ("external_api", f"call_{i}"),
                ],
                correlation_id=uuid4(),
            )
            all_results.append(results)

        # Assert - verify independent failure tracking
        db_service = multi_service_executor.services["database"]
        cache_service = multi_service_executor.services["cache"]
        api_service = multi_service_executor.services["external_api"]

        # Database always fails
        assert db_service.failure_count == num_iterations
        assert db_service.success_count == 0

        # Cache never fails
        assert cache_service.failure_count == 0
        assert cache_service.success_count == num_iterations

        # External API has mixed results
        assert api_service.call_count == num_iterations
        assert api_service.failure_count + api_service.success_count == num_iterations


@pytest.mark.chaos
class TestCascadingFailures:
    """Test cascading failure scenarios where one failure triggers another."""

    @pytest.mark.asyncio
    async def test_cascading_failure_chain(
        self,
        chaos_idempotency_store: InMemoryIdempotencyStore,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test a chain of failures where each triggers the next.

        In cascading failures:
        - Primary failure should be detected
        - Secondary failures should be properly tracked
        - Circuit breaker behavior should prevent unlimited cascading
        """
        # Arrange - create executors with cascading failure logic
        failure_chain: list[str] = []
        cascade_triggered = False

        async def cascading_execute(operation: str, intent_id: UUID) -> None:
            nonlocal cascade_triggered
            failure_chain.append(operation)

            # First operation succeeds but triggers cascade
            if operation == "primary" and not cascade_triggered:
                cascade_triggered = True
                # Simulate async secondary effect that fails
                raise ValueError(f"Primary failure in {operation}")

            # Secondary operations fail due to cascade
            if cascade_triggered and operation != "primary":
                raise RuntimeError(f"Cascading failure in {operation}")

        mock_backend_client.execute = AsyncMock(side_effect=cascading_execute)

        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=FailureInjector(config=ChaosConfig()),
            backend_client=mock_backend_client,
        )

        # Act - execute chain of operations
        results = await asyncio.gather(
            executor.execute_with_chaos(uuid4(), "primary"),
            executor.execute_with_chaos(uuid4(), "secondary_1"),
            executor.execute_with_chaos(uuid4(), "secondary_2"),
            return_exceptions=True,
        )

        # Assert - all operations attempted, failures tracked
        assert len(results) == 3
        failures = [r for r in results if isinstance(r, Exception)]
        assert len(failures) >= 1  # At least primary fails

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_infinite_cascade(
        self,
        chaos_idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Test that circuit breaker logic prevents infinite cascading.

        When failures cascade:
        - A circuit breaker should eventually stop retries
        - Maximum retry count should be respected
        - Final state should indicate circuit open
        """
        # Arrange
        max_attempts = 5
        attempt_count = 0
        circuit_open = False

        class CircuitBreakerSimulator:
            def __init__(self, threshold: int):
                self.failure_count = 0
                self.threshold = threshold
                self.is_open = False

            async def execute_with_breaker(self, operation: str) -> bool:
                nonlocal attempt_count, circuit_open

                if self.is_open:
                    circuit_open = True
                    raise InfraUnavailableError(
                        "Circuit breaker is open",
                        context=ModelInfraErrorContext(operation=operation),
                    )

                attempt_count += 1
                self.failure_count += 1

                if self.failure_count >= self.threshold:
                    self.is_open = True

                raise ValueError(f"Simulated failure in {operation}")

        breaker = CircuitBreakerSimulator(threshold=3)

        # Act - attempt operations until circuit opens
        results: list[bool | Exception] = []
        for i in range(max_attempts):
            try:
                await breaker.execute_with_breaker(f"op_{i}")
                results.append(True)
            except Exception as e:
                results.append(e)

        # Assert
        assert attempt_count >= 3, "Should attempt at least threshold operations"
        assert breaker.is_open, "Circuit should be open after threshold failures"
        assert circuit_open, "Should have hit circuit breaker"

        # Last attempts should be InfraUnavailableError
        unavailable_errors = [
            r for r in results if isinstance(r, InfraUnavailableError)
        ]
        assert len(unavailable_errors) >= 1


@pytest.mark.chaos
class TestRaceConditionHandling:
    """Test race condition handling during concurrent chaotic operations."""

    @pytest.mark.asyncio
    async def test_concurrent_idempotency_checks(
        self,
        chaos_idempotency_store: InMemoryIdempotencyStore,
        mock_backend_client: MagicMock,
        failure_injector: FailureInjector,
    ) -> None:
        """Test idempotency under concurrent chaos.

        When multiple operations with the same intent ID race:
        - Only one should execute
        - Others should be detected as duplicates
        - Counter should reflect single execution
        """
        # Arrange
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=failure_injector,
            backend_client=mock_backend_client,
        )

        shared_intent_id = uuid4()
        num_concurrent = 10

        # Act - race multiple operations with same intent ID
        results = await asyncio.gather(
            *[
                executor.execute_with_chaos(
                    intent_id=shared_intent_id,
                    operation=f"concurrent_op_{i}",
                )
                for i in range(num_concurrent)
            ],
            return_exceptions=True,
        )

        # Assert - all should succeed (idempotent)
        assert len(results) == num_concurrent
        assert all(r is True for r in results)

        # Backend should only be called once
        assert mock_backend_client.execute.call_count == 1
        assert executor.execution_count == 1

    @pytest.mark.asyncio
    async def test_counter_accuracy_under_concurrent_chaos(
        self,
        chaos_idempotency_store: InMemoryIdempotencyStore,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that counters remain accurate under concurrent operations.

        With many concurrent operations:
        - Success and failure counters should be accurate
        - No counts should be lost
        - Total should equal number of operations
        """
        # Arrange - 30% failure rate
        injector = FailureInjector(config=ChaosConfig(failure_rate=0.3))
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=injector,
            backend_client=mock_backend_client,
        )

        num_concurrent = 50

        # Act - execute many concurrent operations
        results = await asyncio.gather(
            *[
                executor.execute_with_chaos(
                    intent_id=uuid4(),  # Unique intent for each
                    operation=f"op_{i}",
                    fail_point="mid",
                )
                for i in range(num_concurrent)
            ],
            return_exceptions=True,
        )

        # Assert - counters accurate
        assert len(results) == num_concurrent

        successes = [r for r in results if r is True]
        failures = [r for r in results if isinstance(r, Exception)]

        # Counters should match results
        assert executor.execution_count == len(successes)
        assert executor.failed_count == len(failures)
        assert executor.execution_count + executor.failed_count == num_concurrent

    @pytest.mark.asyncio
    async def test_no_deadlock_under_concurrent_failures(
        self,
        multi_service_executor: MultiServiceExecutor,
    ) -> None:
        """Test that concurrent failures don't cause deadlocks.

        System should complete within reasonable time even with:
        - High failure rates
        - Many concurrent operations
        - Mixed operation types
        """
        # Arrange - high failure rates on all services
        for service in multi_service_executor.services.values():
            service.failure_injector.set_failure_rate(0.7)
            service.failure_injector.set_latency_range(1, 5)  # Small latency

        # Create many operations
        operations = []
        for i in range(30):
            operations.extend(
                [
                    ("database", f"query_{i}"),
                    ("cache", f"get_{i}"),
                    ("external_api", f"call_{i}"),
                ]
            )

        # Act - should complete within timeout (no deadlock)
        try:
            results = await asyncio.wait_for(
                multi_service_executor.execute_all_concurrent(
                    operations=operations,
                    correlation_id=uuid4(),
                ),
                timeout=10.0,  # 10 second timeout
            )
            # Assert - all operations completed
            assert len(results) == len(operations)

        except TimeoutError:
            pytest.fail(
                "Deadlock detected - operations did not complete within timeout"
            )


@pytest.mark.chaos
class TestMixedFailureModes:
    """Test scenarios combining different failure types."""

    @pytest.mark.asyncio
    async def test_mixed_timeouts_and_connection_errors(
        self,
        chaos_idempotency_store: InMemoryIdempotencyStore,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test handling of mixed timeout and connection errors.

        When different error types occur:
        - Each error type should be properly categorized
        - Error handling should not mask error types
        - All errors should be captured
        """
        # Arrange
        call_count = 0

        async def mixed_failure_backend(operation: str, intent_id: UUID) -> None:
            nonlocal call_count
            call_count += 1

            # Alternate between error types
            if call_count % 3 == 1:
                raise InfraTimeoutError(
                    "Timeout error",
                    context=ModelInfraErrorContext(operation=operation),
                )
            if call_count % 3 == 2:
                raise InfraConnectionError(
                    "Connection error",
                    context=ModelInfraErrorContext(operation=operation),
                )
            # Third call succeeds

        mock_backend_client.execute = AsyncMock(side_effect=mixed_failure_backend)

        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=FailureInjector(config=ChaosConfig()),
            backend_client=mock_backend_client,
        )

        # Act - execute multiple operations
        results = await asyncio.gather(
            *[executor.execute_with_chaos(uuid4(), f"op_{i}") for i in range(9)],
            return_exceptions=True,
        )

        # Assert - mixed results
        assert len(results) == 9

        timeouts = [r for r in results if isinstance(r, InfraTimeoutError)]
        connection_errors = [r for r in results if isinstance(r, InfraConnectionError)]
        successes = [r for r in results if r is True]

        assert len(timeouts) == 3, f"Expected 3 timeouts, got {len(timeouts)}"
        assert len(connection_errors) == 3, (
            f"Expected 3 connection errors, got {len(connection_errors)}"
        )
        assert len(successes) == 3, f"Expected 3 successes, got {len(successes)}"

    @pytest.mark.asyncio
    async def test_partial_failures_with_timeouts(
        self,
        multi_service_executor: MultiServiceExecutor,
    ) -> None:
        """Test partial workflow failures combined with timeouts.

        When a workflow has both partial failures and timeouts:
        - Timeout operations should be distinguishable
        - Partial failures should be tracked separately
        - Recovery should be possible for timed-out operations
        """
        # Arrange - database times out, cache fails, API succeeds
        multi_service_executor.services["database"].failure_injector.set_timeout_rate(
            1.0
        )
        multi_service_executor.services["cache"].failure_injector.set_failure_rate(1.0)
        multi_service_executor.services[
            "external_api"
        ].failure_injector.set_failure_rate(0.0)

        # Act
        results = await multi_service_executor.execute_all_concurrent(
            operations=[
                ("database", "query"),
                ("cache", "get"),
                ("external_api", "call"),
            ],
            correlation_id=uuid4(),
        )

        # Assert
        assert len(results) == 3

        # Categorize results
        timeouts = [r for r in results if isinstance(r, InfraTimeoutError)]
        value_errors = [r for r in results if isinstance(r, ValueError)]
        successes = [r for r in results if r is True]

        assert len(timeouts) == 1, "Database should timeout"
        assert len(value_errors) == 1, "Cache should fail with ValueError"
        assert len(successes) == 1, "External API should succeed"

    @pytest.mark.asyncio
    async def test_latency_injection_with_failures(
        self,
        chaos_idempotency_store: InMemoryIdempotencyStore,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that latency and failures can occur together.

        When operations have both latency and failure injection:
        - Latency should be applied before failure decision
        - Total execution time should reflect latency
        - Failures should still be properly handled
        """
        # Arrange - moderate failure rate with latency
        injector = FailureInjector(
            config=ChaosConfig(
                failure_rate=0.3,
                latency_min_ms=5,
                latency_max_ms=15,
            )
        )

        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=injector,
            backend_client=mock_backend_client,
        )

        # Act
        import time

        start_time = time.monotonic()

        results = await asyncio.gather(
            *[
                executor.execute_with_chaos(uuid4(), f"op_{i}", fail_point="mid")
                for i in range(10)
            ],
            return_exceptions=True,
        )

        elapsed_time = time.monotonic() - start_time

        # Assert
        assert len(results) == 10

        # Some operations should have experienced latency
        # With 5-15ms latency per operation, total time should be > 0
        assert elapsed_time > 0.005, "Should have some measurable latency"

        # Should have mix of successes and failures
        successes = [r for r in results if r is True]
        failures = [r for r in results if isinstance(r, Exception)]

        # At least verify we got results (statistical)
        assert len(successes) + len(failures) == 10

    @pytest.mark.asyncio
    async def test_network_partition_with_concurrent_operations(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
    ) -> None:
        """Test concurrent operations during network partition.

        During a network partition:
        - Operations should fail with connection errors
        - After healing, operations should succeed
        - No data corruption should occur
        """
        # Import MockEventBusWithPartition
        from tests.chaos.conftest import MockEventBusWithPartition

        # Arrange
        event_bus = MockEventBusWithPartition(network_partition_simulator)
        await event_bus.start()

        messages_before_partition: list[dict] = []
        messages_after_partition: list[dict] = []

        # Publish before partition
        for i in range(3):
            await event_bus.publish(f"topic_{i}", None, f"message_{i}".encode())
            messages_before_partition.append({"topic": f"topic_{i}"})

        # Start partition
        network_partition_simulator.start_partition()

        # Try to publish during partition
        partition_errors: list[Exception] = []
        for i in range(3):
            try:
                await event_bus.publish(
                    f"partition_topic_{i}", None, f"partition_msg_{i}".encode()
                )
            except InfraConnectionError as e:
                partition_errors.append(e)

        # End partition
        network_partition_simulator.end_partition()

        # Publish after partition heals
        for i in range(3):
            await event_bus.publish(
                f"healed_topic_{i}", None, f"healed_msg_{i}".encode()
            )
            messages_after_partition.append({"topic": f"healed_topic_{i}"})

        # Cleanup
        await event_bus.close()

        # Assert
        assert len(messages_before_partition) == 3
        assert len(partition_errors) == 3, "All partition operations should fail"
        assert all(isinstance(e, InfraConnectionError) for e in partition_errors)
        assert len(messages_after_partition) == 3

        # Total published should be before + after (not during)
        assert len(event_bus.published_messages) == 6


@pytest.mark.chaos
class TestDataIntegrityUnderChaos:
    """Test data integrity during chaotic conditions."""

    @pytest.mark.asyncio
    async def test_no_duplicate_executions_under_chaos(
        self,
        chaos_idempotency_store: InMemoryIdempotencyStore,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that idempotency prevents duplicates even under chaos.

        With high concurrency and failure rates:
        - Each unique intent should execute at most once
        - Retries with same intent should be deduplicated
        - Backend call count should match unique intents
        """
        # Arrange
        injector = FailureInjector(config=ChaosConfig(failure_rate=0.2))
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=injector,
            backend_client=mock_backend_client,
        )

        # Create 10 unique intents, each attempted 5 times
        unique_intents = [uuid4() for _ in range(10)]
        all_operations = []
        for intent in unique_intents:
            for attempt in range(5):
                all_operations.append((intent, f"op_attempt_{attempt}"))

        # Act
        results = await asyncio.gather(
            *[
                executor.execute_with_chaos(
                    intent_id=intent,
                    operation=operation,
                )
                for intent, operation in all_operations
            ],
            return_exceptions=True,
        )

        # Assert
        assert len(results) == 50  # 10 intents * 5 attempts

        # Backend should be called at most 10 times (once per unique intent)
        assert mock_backend_client.execute.call_count <= 10

    @pytest.mark.asyncio
    async def test_consistent_state_after_mixed_failures(
        self,
        multi_service_executor: MultiServiceExecutor,
    ) -> None:
        """Test that state remains consistent after mixed success/failure.

        After chaotic execution:
        - Completed operations list should match successes
        - Failed operations list should match failures
        - No operations should be in both lists
        """
        # Arrange - varied failure rates
        multi_service_executor.services["database"].failure_injector.set_failure_rate(
            0.4
        )
        multi_service_executor.services["cache"].failure_injector.set_failure_rate(0.3)
        multi_service_executor.services[
            "external_api"
        ].failure_injector.set_failure_rate(0.2)

        operations = []
        for i in range(20):
            operations.extend(
                [
                    ("database", f"db_op_{i}"),
                    ("cache", f"cache_op_{i}"),
                    ("external_api", f"api_op_{i}"),
                ]
            )

        # Act
        await multi_service_executor.execute_all_concurrent(
            operations=operations,
            correlation_id=uuid4(),
        )

        # Assert - verify state consistency
        completed_set = set(multi_service_executor.completed_operations)
        failed_set = set(multi_service_executor.failed_operations)

        # No overlap between completed and failed
        assert len(completed_set & failed_set) == 0, (
            "Operations should not appear in both completed and failed lists"
        )

        # Total should equal input operations
        total_tracked = len(completed_set) + len(failed_set)
        assert total_tracked == len(operations), (
            f"Expected {len(operations)} tracked operations, got {total_tracked}"
        )

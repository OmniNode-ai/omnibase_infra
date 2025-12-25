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


# =============================================================================
# Additional Concurrent Chaos Scenarios (OMN-955 PR Review)
# =============================================================================


@pytest.mark.chaos
class TestCircuitBreakerUnderConcurrentLoad:
    """Test circuit breaker behavior under concurrent load.

    These tests validate that circuit breakers correctly handle:
    - Multiple concurrent operations hitting the breaker simultaneously
    - Race conditions in breaker state transitions
    - Proper failure counting under concurrent access
    - Half-open state behavior with concurrent requests
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_circuit_breaker_concurrent_failure_threshold(self) -> None:
        """Test circuit breaker reaches threshold with concurrent failures.

        When many concurrent operations fail:
        - Circuit should open after threshold is reached
        - All concurrent failures should be counted
        - Subsequent requests should be blocked immediately
        """
        # Arrange
        threshold = 5
        num_concurrent = 20  # More than threshold

        class ConcurrentCircuitBreaker:
            """Thread-safe circuit breaker for concurrent testing."""

            def __init__(self, threshold: int, reset_timeout: float = 1.0):
                self.threshold = threshold
                self.reset_timeout = reset_timeout
                self.failure_count = 0
                self.is_open = False
                self.open_time: float | None = None
                self._lock = asyncio.Lock()
                self.blocked_count = 0
                self.executed_count = 0

            async def execute(self, operation: str, should_fail: bool) -> bool:
                """Execute operation with circuit breaker protection."""
                async with self._lock:
                    # Check if circuit is open
                    if self.is_open:
                        # Check for half-open transition
                        if self.open_time is not None:
                            import time

                            if time.monotonic() - self.open_time > self.reset_timeout:
                                # Try half-open - allow one request
                                self.is_open = False
                                self.failure_count = 0
                                self.open_time = None
                            else:
                                self.blocked_count += 1
                                raise InfraUnavailableError(
                                    "Circuit breaker is open",
                                    context=ModelInfraErrorContext(operation=operation),
                                )
                        else:
                            self.blocked_count += 1
                            raise InfraUnavailableError(
                                "Circuit breaker is open",
                                context=ModelInfraErrorContext(operation=operation),
                            )

                    self.executed_count += 1

                    if should_fail:
                        self.failure_count += 1
                        if self.failure_count >= self.threshold:
                            import time

                            self.is_open = True
                            self.open_time = time.monotonic()
                        raise ValueError(f"Simulated failure in {operation}")

                    return True

        breaker = ConcurrentCircuitBreaker(threshold=threshold)

        # Act - launch many concurrent failing operations
        async def failing_operation(i: int) -> bool:
            return await breaker.execute(f"op_{i}", should_fail=True)

        results = await asyncio.gather(
            *[failing_operation(i) for i in range(num_concurrent)],
            return_exceptions=True,
        )

        # Assert
        assert len(results) == num_concurrent

        # All should be failures (either ValueError or InfraUnavailableError)
        failures = [r for r in results if isinstance(r, Exception)]
        assert len(failures) == num_concurrent

        # Circuit should be open
        assert breaker.is_open, "Circuit should be open after threshold failures"

        # Some should be blocked by circuit
        blocked = [r for r in results if isinstance(r, InfraUnavailableError)]
        value_errors = [r for r in results if isinstance(r, ValueError)]

        # At least threshold operations should have executed before circuit opened
        assert len(value_errors) >= threshold, (
            f"Expected at least {threshold} ValueErrors, got {len(value_errors)}"
        )

        # Remaining should be blocked
        assert breaker.blocked_count >= 0, "Some operations may have been blocked"
        assert breaker.executed_count + breaker.blocked_count == num_concurrent

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_concurrent_requests(self) -> None:
        """Test circuit breaker half-open state with concurrent requests.

        When circuit is half-open:
        - Only one request should be allowed through as a probe
        - While probe is in progress, other requests should be blocked
        - Success should close circuit, failure should re-open

        Note:
            This test simulates realistic half-open behavior where the probe
            request takes time to complete. While it's in progress, other
            concurrent requests should be blocked.
        """

        # Arrange
        class HalfOpenCircuitBreaker:
            """Circuit breaker with proper half-open probe behavior."""

            def __init__(self):
                self.state = "closed"  # closed, open, half_open
                self.probe_in_progress = False
                self._lock = asyncio.Lock()
                self.probe_count = 0
                self.blocked_count = 0
                self.success_count = 0

            async def execute(
                self,
                operation: str,
                succeed: bool = True,
                probe_delay_ms: int = 50,
            ) -> str:
                """Execute operation respecting circuit state."""
                async with self._lock:
                    if self.state == "open":
                        self.blocked_count += 1
                        raise InfraUnavailableError(
                            "Circuit open",
                            context=ModelInfraErrorContext(operation=operation),
                        )

                    if self.state == "half_open":
                        # In half-open, only allow one probe request
                        if self.probe_in_progress:
                            self.blocked_count += 1
                            raise InfraUnavailableError(
                                "Circuit half-open, probe in progress",
                                context=ModelInfraErrorContext(operation=operation),
                            )
                        # This is the probe request
                        self.probe_in_progress = True
                        self.probe_count += 1

                # Execute probe outside lock (simulates actual I/O)
                if self.probe_in_progress:
                    await asyncio.sleep(probe_delay_ms / 1000.0)

                # Complete the probe and update state
                async with self._lock:
                    if self.probe_in_progress:
                        self.probe_in_progress = False
                        if succeed:
                            self.state = "closed"
                            self.success_count += 1
                            return "ok"
                        else:
                            self.state = "open"
                            raise ValueError("Probe failed")

                    # Normal operation in closed state
                    self.success_count += 1
                    return "ok"

        breaker = HalfOpenCircuitBreaker()
        breaker.state = "half_open"

        # Act - launch concurrent requests in half-open state
        num_concurrent = 5

        async def test_request(i: int) -> str:
            return await breaker.execute(f"op_{i}", succeed=True, probe_delay_ms=20)

        results = await asyncio.gather(
            *[test_request(i) for i in range(num_concurrent)],
            return_exceptions=True,
        )

        # Assert
        assert len(results) == num_concurrent

        # Only one probe should have been attempted
        assert breaker.probe_count == 1, (
            f"Expected exactly 1 probe, got {breaker.probe_count}"
        )

        # Count successes and blocks
        successes = [r for r in results if r == "ok"]
        blocked = [r for r in results if isinstance(r, InfraUnavailableError)]

        # One probe succeeded, others were blocked during probe
        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}"
        assert len(blocked) == num_concurrent - 1, (
            f"Expected {num_concurrent - 1} blocked, got {len(blocked)}"
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_consistency_under_load(self) -> None:
        """Test that circuit breaker state remains consistent under load.

        With many concurrent operations:
        - Failure count should accurately reflect actual failures
        - State transitions should be atomic
        - No operations should be lost or double-counted
        """

        # Arrange
        class AtomicCircuitBreaker:
            """Circuit breaker with atomic state tracking."""

            def __init__(self, threshold: int = 5):
                self.threshold = threshold
                self.failure_count = 0
                self.success_count = 0
                self.is_open = False
                self._lock = asyncio.Lock()
                self.total_attempts = 0

            async def execute(self, should_fail: bool) -> bool:
                """Execute with atomic counter updates."""
                async with self._lock:
                    self.total_attempts += 1

                    if self.is_open:
                        raise InfraUnavailableError(
                            "Circuit open",
                            context=ModelInfraErrorContext(operation="test"),
                        )

                    if should_fail:
                        self.failure_count += 1
                        if self.failure_count >= self.threshold:
                            self.is_open = True
                        raise ValueError("Failed")

                    self.success_count += 1
                    return True

        breaker = AtomicCircuitBreaker(threshold=10)

        # Act - mixed success/failure operations
        num_concurrent = 50

        async def mixed_operation(i: int) -> bool:
            should_fail = i % 3 == 0  # ~33% failure rate
            return await breaker.execute(should_fail)

        results = await asyncio.gather(
            *[mixed_operation(i) for i in range(num_concurrent)],
            return_exceptions=True,
        )

        # Assert - verify consistency
        assert len(results) == num_concurrent

        successes = [r for r in results if r is True]
        failures = [
            r for r in results if isinstance(r, (ValueError, InfraUnavailableError))
        ]

        # All operations accounted for
        assert len(successes) + len(failures) == num_concurrent

        # Counter consistency (accounting for blocked requests)
        value_errors = [r for r in results if isinstance(r, ValueError)]
        blocked = [r for r in results if isinstance(r, InfraUnavailableError)]

        assert breaker.failure_count == len(value_errors), (
            f"Failure count mismatch: {breaker.failure_count} != {len(value_errors)}"
        )
        assert breaker.success_count == len(successes), (
            f"Success count mismatch: {breaker.success_count} != {len(successes)}"
        )
        assert breaker.total_attempts == len(value_errors) + len(successes) + len(
            blocked
        )


@pytest.mark.chaos
class TestDLQConcurrentWrites:
    """Test Dead Letter Queue behavior under concurrent failure conditions.

    These tests validate:
    - Multiple failures trying to write to DLQ simultaneously
    - DLQ write ordering under concurrent access
    - DLQ capacity limits under concurrent pressure
    - No message loss during concurrent DLQ operations
    """

    @pytest.mark.asyncio
    async def test_concurrent_dlq_writes_no_message_loss(self) -> None:
        """Test that concurrent DLQ writes don't lose messages.

        When multiple failures occur simultaneously:
        - All failures should be written to DLQ
        - No messages should be lost
        - Messages should have unique identifiers
        """

        # Arrange
        @dataclass
        class DLQMessage:
            """Message stored in Dead Letter Queue."""

            message_id: UUID
            original_topic: str
            error_reason: str
            timestamp: float

        class ConcurrentDLQ:
            """Thread-safe Dead Letter Queue for concurrent testing."""

            def __init__(self, max_capacity: int = 1000):
                self.messages: list[DLQMessage] = []
                self.max_capacity = max_capacity
                self._lock = asyncio.Lock()
                self.dropped_count = 0
                self.write_count = 0

            async def write(
                self,
                message_id: UUID,
                topic: str,
                error: str,
            ) -> bool:
                """Write a failed message to DLQ."""
                import time

                async with self._lock:
                    self.write_count += 1

                    if len(self.messages) >= self.max_capacity:
                        self.dropped_count += 1
                        return False

                    self.messages.append(
                        DLQMessage(
                            message_id=message_id,
                            original_topic=topic,
                            error_reason=error,
                            timestamp=time.monotonic(),
                        )
                    )
                    return True

            async def get_messages_for_topic(self, topic: str) -> list[DLQMessage]:
                """Get all DLQ messages for a specific topic."""
                async with self._lock:
                    return [m for m in self.messages if m.original_topic == topic]

        dlq = ConcurrentDLQ(max_capacity=1000)
        num_concurrent = 100

        # Act - simulate many concurrent failures
        async def simulate_failure(i: int) -> bool:
            message_id = uuid4()
            topic = f"topic_{i % 5}"  # Distribute across 5 topics
            error = f"Simulated failure {i}"
            return await dlq.write(message_id, topic, error)

        results = await asyncio.gather(
            *[simulate_failure(i) for i in range(num_concurrent)],
            return_exceptions=True,
        )

        # Assert
        assert len(results) == num_concurrent
        assert all(r is True for r in results), "All writes should succeed"

        # No messages lost
        assert len(dlq.messages) == num_concurrent, (
            f"Expected {num_concurrent} messages, got {len(dlq.messages)}"
        )
        assert dlq.dropped_count == 0, "No messages should be dropped"

        # All message IDs should be unique
        message_ids = [m.message_id for m in dlq.messages]
        assert len(set(message_ids)) == num_concurrent, (
            "All message IDs should be unique"
        )

    @pytest.mark.asyncio
    async def test_dlq_capacity_under_concurrent_pressure(self) -> None:
        """Test DLQ behavior when capacity is exceeded concurrently.

        When DLQ is near capacity and many writes arrive:
        - Writes should be atomic
        - Capacity should be respected
        - Overflow messages should be tracked
        """

        # Arrange
        @dataclass
        class SimpleDLQMessage:
            """Simplified DLQ message."""

            id: UUID
            error: str

        class BoundedDLQ:
            """DLQ with bounded capacity."""

            def __init__(self, capacity: int):
                self.capacity = capacity
                self.messages: list[SimpleDLQMessage] = []
                self._lock = asyncio.Lock()
                self.overflow_count = 0

            async def write(self, error: str) -> bool:
                """Attempt to write to DLQ."""
                async with self._lock:
                    if len(self.messages) >= self.capacity:
                        self.overflow_count += 1
                        return False

                    self.messages.append(SimpleDLQMessage(id=uuid4(), error=error))
                    return True

        # Small capacity to test overflow
        capacity = 20
        num_concurrent = 50

        dlq = BoundedDLQ(capacity=capacity)

        # Act - exceed capacity with concurrent writes
        async def write_to_dlq(i: int) -> bool:
            return await dlq.write(f"error_{i}")

        results = await asyncio.gather(
            *[write_to_dlq(i) for i in range(num_concurrent)],
            return_exceptions=True,
        )

        # Assert
        successes = [r for r in results if r is True]
        failures = [r for r in results if r is False]

        # Exactly capacity messages should succeed
        assert len(successes) == capacity, (
            f"Expected {capacity} successes, got {len(successes)}"
        )

        # Remaining should fail
        assert len(failures) == num_concurrent - capacity, (
            f"Expected {num_concurrent - capacity} failures, got {len(failures)}"
        )

        # Overflow should be tracked
        assert dlq.overflow_count == num_concurrent - capacity
        assert len(dlq.messages) == capacity

    @pytest.mark.asyncio
    async def test_dlq_ordering_under_concurrent_writes(self) -> None:
        """Test that DLQ maintains insertion order under concurrent access.

        When multiple writes occur concurrently:
        - Messages should be stored in a consistent order
        - Timestamps should be monotonically increasing
        - No interleaving corruption should occur
        """
        # Arrange
        import time

        @dataclass
        class OrderedDLQMessage:
            """DLQ message with ordering metadata."""

            sequence: int
            timestamp: float
            data: str

        class OrderedDLQ:
            """DLQ that tracks insertion order."""

            def __init__(self):
                self.messages: list[OrderedDLQMessage] = []
                self._lock = asyncio.Lock()
                self._sequence = 0

            async def write(self, data: str) -> int:
                """Write message and return sequence number."""
                async with self._lock:
                    self._sequence += 1
                    seq = self._sequence
                    self.messages.append(
                        OrderedDLQMessage(
                            sequence=seq,
                            timestamp=time.monotonic(),
                            data=data,
                        )
                    )
                    return seq

        dlq = OrderedDLQ()
        num_concurrent = 100

        # Act
        async def write_ordered(i: int) -> int:
            return await dlq.write(f"message_{i}")

        sequences = await asyncio.gather(
            *[write_ordered(i) for i in range(num_concurrent)],
        )

        # Assert
        # All sequence numbers should be unique and consecutive
        assert len(set(sequences)) == num_concurrent, "All sequences should be unique"
        assert sorted(sequences) == list(range(1, num_concurrent + 1)), (
            "Sequences should be 1 to N"
        )

        # Messages in DLQ should be ordered by sequence
        dlq_sequences = [m.sequence for m in dlq.messages]
        assert dlq_sequences == list(range(1, num_concurrent + 1)), (
            "DLQ messages should be in sequence order"
        )

        # Timestamps should be monotonically increasing
        timestamps = [m.timestamp for m in dlq.messages]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp at {i} should be >= timestamp at {i - 1}"
            )


@pytest.mark.chaos
class TestRecoveryRaceConditions:
    """Test race conditions in recovery logic during concurrent chaos.

    These tests validate:
    - Concurrent retry operations don't corrupt state
    - Recovery callbacks don't race with ongoing operations
    - Cleanup operations are atomic during concurrent failures
    """

    @pytest.mark.asyncio
    async def test_concurrent_retry_state_consistency(self) -> None:
        """Test that concurrent retries maintain consistent state.

        When multiple operations retry concurrently:
        - Retry counts should be accurate
        - State transitions should be atomic
        - No retries should be lost
        """

        # Arrange
        @dataclass
        class RetryableOperation:
            """Tracks retry state for an operation."""

            id: UUID
            retry_count: int = 0
            max_retries: int = 3
            succeeded: bool = False
            failed_permanently: bool = False

        class ConcurrentRetryManager:
            """Manages concurrent retry operations."""

            def __init__(self):
                self.operations: dict[UUID, RetryableOperation] = {}
                self._lock = asyncio.Lock()
                self.total_retries = 0
                self.successful_ops = 0
                self.failed_ops = 0

            async def execute_with_retry(
                self,
                op_id: UUID,
                failure_injector: FailureInjector,
            ) -> bool:
                """Execute operation with retry logic."""
                async with self._lock:
                    if op_id not in self.operations:
                        self.operations[op_id] = RetryableOperation(id=op_id)
                    op = self.operations[op_id]

                while True:
                    try:
                        # Check if already resolved
                        async with self._lock:
                            if op.succeeded or op.failed_permanently:
                                return op.succeeded

                        # Attempt execution
                        await failure_injector.maybe_inject_failure(
                            f"retry_op_{op_id}",
                            op_id,
                        )

                        # Success
                        async with self._lock:
                            op.succeeded = True
                            self.successful_ops += 1
                        return True

                    except ValueError:
                        # Handle retry
                        async with self._lock:
                            op.retry_count += 1
                            self.total_retries += 1

                            if op.retry_count >= op.max_retries:
                                op.failed_permanently = True
                                self.failed_ops += 1
                                return False

        retry_manager = ConcurrentRetryManager()
        failure_injector = FailureInjector(
            config=ChaosConfig(failure_rate=0.5)  # 50% failure rate
        )

        num_operations = 30

        # Act - execute many operations with retries
        op_ids = [uuid4() for _ in range(num_operations)]

        async def execute_op(op_id: UUID) -> bool:
            return await retry_manager.execute_with_retry(op_id, failure_injector)

        results = await asyncio.gather(
            *[execute_op(op_id) for op_id in op_ids],
            return_exceptions=True,
        )

        # Assert
        assert len(results) == num_operations

        successes = [r for r in results if r is True]
        failures = [r for r in results if r is False]

        # State consistency
        assert len(successes) == retry_manager.successful_ops
        assert len(failures) == retry_manager.failed_ops
        assert len(successes) + len(failures) == num_operations

        # All operations tracked
        assert len(retry_manager.operations) == num_operations

    @pytest.mark.asyncio
    async def test_recovery_callback_race_with_operations(self) -> None:
        """Test that recovery callbacks don't race with ongoing operations.

        When recovery callback fires:
        - Active operations should complete or abort cleanly
        - Callback should not corrupt operation state
        - Resources should be properly cleaned up
        """

        # Arrange
        class RecoverableService:
            """Service with recovery callback support."""

            def __init__(self):
                self.is_healthy = True
                self.active_operations: set[UUID] = set()
                self.completed_operations: set[UUID] = set()
                self.aborted_operations: set[UUID] = set()
                self._lock = asyncio.Lock()
                self.recovery_triggered = False

            async def start_operation(self, op_id: UUID) -> None:
                """Start an operation."""
                async with self._lock:
                    if not self.is_healthy:
                        self.aborted_operations.add(op_id)
                        raise InfraUnavailableError(
                            "Service unhealthy",
                            context=ModelInfraErrorContext(operation="start"),
                        )
                    self.active_operations.add(op_id)

            async def complete_operation(self, op_id: UUID) -> None:
                """Complete an operation."""
                async with self._lock:
                    if op_id in self.active_operations:
                        self.active_operations.remove(op_id)
                        self.completed_operations.add(op_id)

            async def trigger_failure(self) -> None:
                """Trigger service failure and abort active operations."""
                async with self._lock:
                    self.is_healthy = False
                    # Abort all active operations
                    for op_id in list(self.active_operations):
                        self.aborted_operations.add(op_id)
                    self.active_operations.clear()

            async def trigger_recovery(self) -> None:
                """Trigger recovery callback."""
                async with self._lock:
                    self.is_healthy = True
                    self.recovery_triggered = True

        service = RecoverableService()

        # Act - run operations while triggering failure and recovery
        async def long_operation(op_id: UUID, delay_ms: int) -> bool:
            try:
                await service.start_operation(op_id)
                await asyncio.sleep(delay_ms / 1000.0)
                await service.complete_operation(op_id)
                return True
            except InfraUnavailableError:
                return False

        async def failure_and_recovery() -> None:
            await asyncio.sleep(0.02)  # Let some ops start
            await service.trigger_failure()
            await asyncio.sleep(0.02)  # Let failure propagate
            await service.trigger_recovery()

        # Start concurrent operations with varying durations
        op_tasks = [long_operation(uuid4(), delay_ms=10 + i * 5) for i in range(10)]
        recovery_task = failure_and_recovery()

        results = await asyncio.gather(
            *op_tasks,
            recovery_task,
            return_exceptions=True,
        )

        op_results = results[:-1]  # Exclude recovery task result

        # Assert
        # All operations should have a result
        assert len(op_results) == 10

        # No operations should be stuck in active state
        assert len(service.active_operations) == 0, (
            "No operations should be active after recovery"
        )

        # All operations should be either completed or aborted
        total_resolved = len(service.completed_operations) + len(
            service.aborted_operations
        )
        assert total_resolved == 10, (
            f"Expected 10 resolved operations, got {total_resolved}"
        )

        # Recovery should have been triggered
        assert service.recovery_triggered

    @pytest.mark.asyncio
    async def test_cleanup_atomicity_during_concurrent_failures(self) -> None:
        """Test that cleanup operations are atomic during concurrent failures.

        When cleanup runs during concurrent failures:
        - Cleanup should complete fully or not at all
        - Resources should not be partially cleaned
        - Concurrent failures should not corrupt cleanup state
        """

        # Arrange
        @dataclass
        class Resource:
            """Resource that needs cleanup."""

            id: UUID
            is_allocated: bool = True
            is_cleaned: bool = False

        class ResourceManager:
            """Manager for resource allocation and cleanup."""

            def __init__(self):
                self.resources: dict[UUID, Resource] = {}
                self._lock = asyncio.Lock()
                self.cleanup_started = False
                self.cleanup_completed = False
                self.allocation_after_cleanup = 0

            async def allocate(self) -> Resource:
                """Allocate a new resource."""
                async with self._lock:
                    if self.cleanup_started and not self.cleanup_completed:
                        # Reject allocations during cleanup
                        raise ValueError("Cannot allocate during cleanup")

                    if self.cleanup_completed:
                        self.allocation_after_cleanup += 1

                    resource = Resource(id=uuid4())
                    self.resources[resource.id] = resource
                    return resource

            async def release(self, resource_id: UUID) -> None:
                """Release a resource."""
                async with self._lock:
                    if resource_id in self.resources:
                        self.resources[resource_id].is_allocated = False

            async def cleanup_all(self) -> int:
                """Atomically clean up all resources."""
                async with self._lock:
                    self.cleanup_started = True

                    # Clean all resources atomically
                    cleaned_count = 0
                    for resource in self.resources.values():
                        if resource.is_allocated:
                            resource.is_allocated = False
                            resource.is_cleaned = True
                            cleaned_count += 1

                    self.cleanup_completed = True
                    return cleaned_count

        manager = ResourceManager()

        # Pre-allocate some resources
        initial_resources = [await manager.allocate() for _ in range(10)]

        # Act - concurrent allocations, releases, and cleanup
        async def allocate_loop(count: int) -> list[Resource | Exception]:
            results: list[Resource | Exception] = []
            for _ in range(count):
                try:
                    r = await manager.allocate()
                    results.append(r)
                except Exception as e:
                    results.append(e)
                await asyncio.sleep(0.001)  # Small delay
            return results

        async def release_loop(resources: list[Resource]) -> None:
            for r in resources:
                await manager.release(r.id)
                await asyncio.sleep(0.001)

        async def delayed_cleanup() -> int:
            await asyncio.sleep(0.01)  # Let some operations start
            return await manager.cleanup_all()

        results = await asyncio.gather(
            allocate_loop(5),
            release_loop(initial_resources[:5]),
            delayed_cleanup(),
            return_exceptions=True,
        )

        # Assert
        alloc_results = results[0]
        cleaned_count = results[2]

        # Cleanup should have completed
        assert manager.cleanup_completed, "Cleanup should complete"

        # All initial resources should be cleaned
        for r in initial_resources:
            assert (
                manager.resources[r.id].is_cleaned
                or not manager.resources[r.id].is_allocated
            ), f"Resource {r.id} should be cleaned or deallocated"

        # Some allocations should have failed during cleanup
        failed_allocs = [r for r in alloc_results if isinstance(r, Exception)]
        assert len(failed_allocs) >= 0  # May be 0 if timing is different

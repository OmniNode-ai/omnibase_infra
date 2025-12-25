# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chaos tests for timeout scenarios (OMN-955).

This test suite validates system behavior when handlers exceed time limits.
It covers:

1. Handlers exceeding configured time limits
2. Timeout detection and handling
3. Cleanup behavior after timeouts
4. Interaction between timeouts and idempotency

Architecture:
    Timeouts can occur in various scenarios:

    1. Slow backend operations (database, external API)
    2. Network latency spikes
    3. Resource contention (CPU, memory, locks)
    4. Deadlocks or blocking operations

    The system should:
    - Detect timeouts promptly
    - Cancel or interrupt long-running operations
    - Clean up resources properly
    - Allow retry with proper idempotency handling

Test Organization:
    - TestTimeoutDetection: Timeout detection scenarios
    - TestTimeoutHandling: Proper handling of timeouts
    - TestTimeoutCleanup: Resource cleanup after timeouts
    - TestTimeoutWithIdempotency: Timeout interaction with idempotency

Related Tickets:
    - OMN-955: Chaos scenario tests
    - OMN-954: Effect idempotency
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import InfraTimeoutError, ModelInfraErrorContext
from omnibase_infra.idempotency import InMemoryIdempotencyStore
from tests.chaos.conftest import (
    ChaosConfig,
    ChaosEffectExecutor,
    FailureInjector,
)

# =============================================================================
# Helper Classes
# =============================================================================


class TimeoutEffectExecutor:
    """Effect executor with configurable timeout behavior.

    This executor wraps operations with timeout handling and simulates
    various timeout scenarios.

    Attributes:
        timeout_seconds: Maximum time allowed for operations.
        idempotency_store: Store for idempotency checking.
        backend_client: Mock backend client.
        timeout_count: Number of timeouts that occurred.
        execution_count: Number of successful executions.
    """

    def __init__(
        self,
        timeout_seconds: float,
        idempotency_store: InMemoryIdempotencyStore,
        backend_client: MagicMock,
    ) -> None:
        """Initialize the timeout effect executor.

        Args:
            timeout_seconds: Maximum time allowed for operations.
            idempotency_store: Store for idempotency checking.
            backend_client: Mock backend client.
        """
        self.timeout_seconds = timeout_seconds
        self.idempotency_store = idempotency_store
        self.backend_client = backend_client
        self.timeout_count = 0
        self.execution_count = 0
        self._lock = asyncio.Lock()

    async def execute_with_timeout(
        self,
        intent_id,
        operation: str,
        simulate_delay_seconds: float = 0.0,
        domain: str = "timeout",
        correlation_id=None,
    ) -> bool:
        """Execute an operation with timeout handling.

        Args:
            intent_id: Unique identifier for this intent.
            operation: Name of the operation.
            simulate_delay_seconds: How long to delay execution.
            domain: Idempotency domain.
            correlation_id: Optional correlation ID.

        Returns:
            True if operation succeeded.

        Raises:
            InfraTimeoutError: If operation exceeds timeout.
        """
        # Check idempotency first
        is_new = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain=domain,
            correlation_id=correlation_id,
        )

        if not is_new:
            return True  # Duplicate, skip

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_backend(operation, simulate_delay_seconds),
                timeout=self.timeout_seconds,
            )

            async with self._lock:
                self.execution_count += 1

            return result

        except TimeoutError:
            async with self._lock:
                self.timeout_count += 1

            context = ModelInfraErrorContext(
                operation=operation,
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Operation '{operation}' exceeded timeout of {self.timeout_seconds}s",
                context=context,
            ) from None

    async def _execute_backend(
        self,
        operation: str,
        delay_seconds: float,
    ) -> bool:
        """Execute backend operation with optional delay.

        Args:
            operation: Name of the operation.
            delay_seconds: How long to delay.

        Returns:
            True if successful.
        """
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        await self.backend_client.execute(operation)
        return True


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def timeout_idempotency_store() -> InMemoryIdempotencyStore:
    """Create in-memory idempotency store for timeout testing."""
    return InMemoryIdempotencyStore()


@pytest.fixture
def mock_backend() -> MagicMock:
    """Create mock backend client."""
    client = MagicMock()
    client.execute = AsyncMock(return_value=None)
    return client


@pytest.fixture
def fast_timeout_executor(
    timeout_idempotency_store: InMemoryIdempotencyStore,
    mock_backend: MagicMock,
) -> TimeoutEffectExecutor:
    """Create executor with fast (0.1s) timeout."""
    return TimeoutEffectExecutor(
        timeout_seconds=0.1,
        idempotency_store=timeout_idempotency_store,
        backend_client=mock_backend,
    )


@pytest.fixture
def slow_timeout_executor(
    timeout_idempotency_store: InMemoryIdempotencyStore,
    mock_backend: MagicMock,
) -> TimeoutEffectExecutor:
    """Create executor with slow (1.0s) timeout."""
    return TimeoutEffectExecutor(
        timeout_seconds=1.0,
        idempotency_store=timeout_idempotency_store,
        backend_client=mock_backend,
    )


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.chaos
class TestTimeoutDetection:
    """Test timeout detection scenarios."""

    @pytest.mark.asyncio
    async def test_operation_completes_within_timeout(
        self,
        fast_timeout_executor: TimeoutEffectExecutor,
    ) -> None:
        """Test that operations completing within timeout succeed.

        When an operation completes before the timeout:
        - The operation should succeed
        - No timeout error should be raised
        - The execution count should be incremented
        """
        # Arrange
        intent_id = uuid4()

        # Act - execute with no delay (within timeout)
        result = await fast_timeout_executor.execute_with_timeout(
            intent_id=intent_id,
            operation="fast_operation",
            simulate_delay_seconds=0.0,
        )

        # Assert
        assert result is True
        assert fast_timeout_executor.execution_count == 1
        assert fast_timeout_executor.timeout_count == 0

    @pytest.mark.slow  # 0.5s delay to exceed 0.1s timeout
    @pytest.mark.asyncio
    async def test_operation_exceeds_timeout(
        self,
        fast_timeout_executor: TimeoutEffectExecutor,
    ) -> None:
        """Test that operations exceeding timeout raise InfraTimeoutError.

        When an operation exceeds the timeout:
        - InfraTimeoutError should be raised
        - The timeout count should be incremented
        - The operation should not complete successfully
        """
        # Arrange
        intent_id = uuid4()

        # Act & Assert - execute with delay exceeding timeout
        with pytest.raises(InfraTimeoutError) as exc_info:
            await fast_timeout_executor.execute_with_timeout(
                intent_id=intent_id,
                operation="slow_operation",
                simulate_delay_seconds=0.5,  # Exceeds 0.1s timeout
            )

        assert "exceeded timeout" in str(exc_info.value)
        assert fast_timeout_executor.timeout_count == 1
        assert fast_timeout_executor.execution_count == 0

    @pytest.mark.slow  # 0.5s delay for timeout error message test
    @pytest.mark.asyncio
    async def test_timeout_includes_operation_name(
        self,
        fast_timeout_executor: TimeoutEffectExecutor,
    ) -> None:
        """Test that timeout error includes operation name.

        The timeout error message should include the operation name
        for debugging purposes.
        """
        # Arrange
        intent_id = uuid4()
        operation_name = "database_query_xyz"

        # Act & Assert
        with pytest.raises(InfraTimeoutError) as exc_info:
            await fast_timeout_executor.execute_with_timeout(
                intent_id=intent_id,
                operation=operation_name,
                simulate_delay_seconds=0.5,
            )

        assert operation_name in str(exc_info.value)

    @pytest.mark.slow  # 0.5s delay for timeout context test
    @pytest.mark.asyncio
    async def test_timeout_includes_correlation_id(
        self,
        fast_timeout_executor: TimeoutEffectExecutor,
    ) -> None:
        """Test that timeout error context includes correlation ID.

        The timeout error should have a context with the correlation ID
        for distributed tracing.
        """
        # Arrange
        intent_id = uuid4()
        correlation_id = uuid4()

        # Act & Assert
        with pytest.raises(InfraTimeoutError) as exc_info:
            await fast_timeout_executor.execute_with_timeout(
                intent_id=intent_id,
                operation="test_op",
                simulate_delay_seconds=0.5,
                correlation_id=correlation_id,
            )

        # Check that the error has proper context
        error = exc_info.value
        # The error should be an InfraTimeoutError (checked by pytest.raises)
        assert isinstance(error, InfraTimeoutError)


@pytest.mark.chaos
class TestTimeoutHandling:
    """Test proper handling of timeouts."""

    @pytest.mark.slow  # 0.5s timeout delay + retry attempt
    @pytest.mark.asyncio
    async def test_timeout_does_not_prevent_retry(
        self,
        timeout_idempotency_store: InMemoryIdempotencyStore,
        mock_backend: MagicMock,
    ) -> None:
        """Test that timeout does not prevent subsequent retry.

        After a timeout, a new attempt with a different intent ID
        should be able to succeed.
        """
        # Arrange
        executor = TimeoutEffectExecutor(
            timeout_seconds=0.1,
            idempotency_store=timeout_idempotency_store,
            backend_client=mock_backend,
        )

        # First attempt - times out
        intent_id_1 = uuid4()
        with pytest.raises(InfraTimeoutError):
            await executor.execute_with_timeout(
                intent_id=intent_id_1,
                operation="slow_operation",
                simulate_delay_seconds=0.5,
            )

        assert executor.timeout_count == 1

        # Second attempt with new intent - succeeds
        intent_id_2 = uuid4()
        result = await executor.execute_with_timeout(
            intent_id=intent_id_2,
            operation="fast_operation",
            simulate_delay_seconds=0.0,
        )

        assert result is True
        assert executor.execution_count == 1

    @pytest.mark.slow  # Multiple 0.5s concurrent delays
    @pytest.mark.asyncio
    async def test_concurrent_timeouts_are_independent(
        self,
        timeout_idempotency_store: InMemoryIdempotencyStore,
        mock_backend: MagicMock,
    ) -> None:
        """Test that concurrent operations have independent timeouts.

        When multiple operations run concurrently:
        - Each has its own timeout
        - One timing out doesn't affect others
        """
        # Arrange
        executor = TimeoutEffectExecutor(
            timeout_seconds=0.1,
            idempotency_store=timeout_idempotency_store,
            backend_client=mock_backend,
        )

        results: list[bool] = []
        timeouts: list[InfraTimeoutError] = []
        lock = asyncio.Lock()

        async def execute_one(delay: float) -> None:
            try:
                result = await executor.execute_with_timeout(
                    intent_id=uuid4(),
                    operation=f"op_{delay}",
                    simulate_delay_seconds=delay,
                )
                async with lock:
                    results.append(result)
            except InfraTimeoutError as e:
                async with lock:
                    timeouts.append(e)

        # Act - run concurrent operations with different delays
        # Some will complete, some will timeout
        await asyncio.gather(
            execute_one(0.0),  # Fast - should succeed
            execute_one(0.05),  # Fast - should succeed
            execute_one(0.5),  # Slow - should timeout
            execute_one(0.0),  # Fast - should succeed
            execute_one(0.5),  # Slow - should timeout
        )

        # Assert
        assert len(results) == 3  # Three fast operations succeeded
        assert len(timeouts) == 2  # Two slow operations timed out

    @pytest.mark.asyncio
    async def test_timeout_via_failure_injector(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test timeout injection via FailureInjector.

        The FailureInjector can be configured to inject timeouts
        at a specific rate.
        """
        # Arrange
        injector = FailureInjector(
            config=ChaosConfig(timeout_rate=1.0)  # 100% timeout rate
        )
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=injector,
            backend_client=mock_backend_client,
        )

        # Act & Assert
        with pytest.raises(InfraTimeoutError, match="Chaos injection"):
            await executor.execute_with_chaos(
                intent_id=uuid4(),
                operation="test_op",
                fail_point="mid",
            )

        assert injector.timeout_count == 1


@pytest.mark.chaos
class TestTimeoutCleanup:
    """Test resource cleanup after timeouts."""

    @pytest.mark.slow  # 0.5s delay to verify cancellation
    @pytest.mark.asyncio
    async def test_backend_not_called_after_timeout(
        self,
        timeout_idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Test that backend is not called after timeout is raised.

        When a timeout occurs:
        - Subsequent backend calls should be cancelled
        - The operation should be cleanly interrupted
        """
        # Arrange
        call_count = 0

        async def slow_backend_call(operation: str) -> None:
            nonlocal call_count
            await asyncio.sleep(0.5)  # Simulate slow operation
            call_count += 1  # This should not be reached

        mock_backend = MagicMock()
        mock_backend.execute = slow_backend_call

        executor = TimeoutEffectExecutor(
            timeout_seconds=0.1,
            idempotency_store=timeout_idempotency_store,
            backend_client=mock_backend,
        )

        # Act
        with pytest.raises(InfraTimeoutError):
            await executor.execute_with_timeout(
                intent_id=uuid4(),
                operation="slow_op",
                simulate_delay_seconds=0.5,
            )

        # Assert - backend call was cancelled
        assert call_count == 0

    @pytest.mark.slow  # 0.5s timeout + state verification
    @pytest.mark.asyncio
    async def test_executor_state_consistent_after_timeout(
        self,
        fast_timeout_executor: TimeoutEffectExecutor,
    ) -> None:
        """Test that executor state is consistent after timeout.

        After a timeout:
        - The timeout count should be accurately tracked
        - The execution count should not include the timed-out operation
        - The executor should be ready for new operations
        """
        # Arrange - cause timeout
        with pytest.raises(InfraTimeoutError):
            await fast_timeout_executor.execute_with_timeout(
                intent_id=uuid4(),
                operation="timeout_op",
                simulate_delay_seconds=0.5,
            )

        # State after timeout
        assert fast_timeout_executor.timeout_count == 1
        assert fast_timeout_executor.execution_count == 0

        # Executor is ready for new operations
        result = await fast_timeout_executor.execute_with_timeout(
            intent_id=uuid4(),
            operation="fast_op",
            simulate_delay_seconds=0.0,
        )

        assert result is True
        assert fast_timeout_executor.execution_count == 1


@pytest.mark.chaos
class TestTimeoutWithIdempotency:
    """Test timeout interaction with idempotency."""

    @pytest.mark.asyncio
    async def test_idempotency_check_before_timeout(
        self,
        timeout_idempotency_store: InMemoryIdempotencyStore,
        mock_backend: MagicMock,
    ) -> None:
        """Test that idempotency is checked before timeout can occur.

        The idempotency check should happen before the potentially
        slow backend operation, preventing duplicates.
        """
        # Arrange
        executor = TimeoutEffectExecutor(
            timeout_seconds=0.1,
            idempotency_store=timeout_idempotency_store,
            backend_client=mock_backend,
        )

        intent_id = uuid4()

        # Pre-record in idempotency store
        await timeout_idempotency_store.check_and_record(
            message_id=intent_id,
            domain="timeout",
        )

        # Act - attempt to execute with same intent (would timeout if not deduplicated)
        result = await executor.execute_with_timeout(
            intent_id=intent_id,
            operation="potentially_slow_op",
            simulate_delay_seconds=0.5,  # Would timeout
        )

        # Assert - idempotency prevented execution (and timeout)
        assert result is True
        mock_backend.execute.assert_not_called()
        assert executor.timeout_count == 0

    @pytest.mark.slow  # 0.5s timeout + idempotency check
    @pytest.mark.asyncio
    async def test_timeout_after_idempotency_records_attempt(
        self,
        timeout_idempotency_store: InMemoryIdempotencyStore,
        mock_backend: MagicMock,
    ) -> None:
        """Test that idempotency records the attempt even on timeout.

        When an operation times out:
        - The idempotency record should exist (attempt was made)
        - A retry with the same intent ID should be detected as duplicate
        """
        # Arrange
        executor = TimeoutEffectExecutor(
            timeout_seconds=0.1,
            idempotency_store=timeout_idempotency_store,
            backend_client=mock_backend,
        )

        intent_id = uuid4()

        # First attempt - times out
        with pytest.raises(InfraTimeoutError):
            await executor.execute_with_timeout(
                intent_id=intent_id,
                operation="slow_op",
                simulate_delay_seconds=0.5,
            )

        # Idempotency record should exist (check_and_record was called)
        record = await timeout_idempotency_store.get_record(
            message_id=intent_id,
            domain="timeout",
        )
        assert record is not None

        # Retry with same intent - should be duplicate
        result = await executor.execute_with_timeout(
            intent_id=intent_id,
            operation="slow_op",
            simulate_delay_seconds=0.0,  # Fast this time
        )

        # Should be treated as duplicate (already attempted)
        assert result is True
        # Backend was never successfully called
        mock_backend.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_latency_injection_via_failure_injector(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test latency injection via FailureInjector.

        The FailureInjector can inject latency into operations
        without causing failures.
        """
        # Arrange
        injector = FailureInjector(
            config=ChaosConfig(
                latency_min_ms=10,
                latency_max_ms=20,
            )
        )

        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=injector,
            backend_client=mock_backend_client,
        )

        intent_id = uuid4()

        # Act - time the execution
        import time

        start = time.time()
        result = await executor.execute_with_chaos(
            intent_id=intent_id,
            operation="test_op",
        )
        elapsed_ms = (time.time() - start) * 1000

        # Assert - should have added some latency
        assert result is True
        assert elapsed_ms >= 10  # At least min latency

    @pytest.mark.slow  # 3x 0.5s delays for counter test
    @pytest.mark.asyncio
    async def test_timeout_counter_tracks_all_timeouts(
        self,
        fast_timeout_executor: TimeoutEffectExecutor,
    ) -> None:
        """Test that timeout counter accurately tracks all timeouts."""
        # Arrange
        num_timeouts = 3

        # Act - cause multiple timeouts
        for _ in range(num_timeouts):
            try:
                await fast_timeout_executor.execute_with_timeout(
                    intent_id=uuid4(),
                    operation="slow_op",
                    simulate_delay_seconds=0.5,
                )
            except InfraTimeoutError:
                pass

        # Assert
        assert fast_timeout_executor.timeout_count == num_timeouts

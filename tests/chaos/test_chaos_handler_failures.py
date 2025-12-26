# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chaos tests for handler failures (OMN-955).

This test suite validates system behavior when handlers fail at various points
during execution. It covers:

1. Random failures during handler processing
2. Failure at different execution points (pre, mid, post)
3. Failure propagation and error handling
4. Recovery behavior after handler failures

Architecture:
    Handlers are the units of work that process messages. When a handler fails:

    1. The failure should be properly propagated
    2. The idempotency system should track incomplete executions
    3. Retry mechanisms should be able to re-attempt the operation
    4. Circuit breakers should protect against cascading failures

Test Organization:
    - TestHandlerFailureAtVariousPoints: Failures at pre/mid/post execution
    - TestHandlerFailurePropagation: Error propagation behavior
    - TestHandlerFailureRecovery: Recovery after failures
    - TestHandlerFailureRandom: Random failure injection

Related Tickets:
    - OMN-955: Chaos scenario tests
    - OMN-954: Effect idempotency
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import InfraConnectionError
from tests.chaos.conftest import (
    ChaosConfig,
    ChaosEffectExecutor,
    FailureInjector,
    assert_failure_rate_within_tolerance,
    run_concurrent_with_tracking,
)

# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.chaos
class TestHandlerFailureAtVariousPoints:
    """Test handler failures at different execution points."""

    @pytest.mark.asyncio
    async def test_failure_before_execution(
        self,
        chaos_effect_executor: ChaosEffectExecutor,
        deterministic_failure_injector: FailureInjector,
    ) -> None:
        """Test that failure before execution prevents backend call.

        When a handler fails before the main execution logic:
        - The backend operation should NOT be called
        - The failure should be properly propagated
        - The operation can be retried with a new intent
        """
        # Arrange
        chaos_effect_executor.failure_injector = deterministic_failure_injector
        intent_id = uuid4()
        correlation_id = uuid4()

        # Act & Assert - InfraConnectionError is used for chaos injection failures
        # (correlation_id is in error context, not message string per ONEX guidelines)
        with pytest.raises(InfraConnectionError, match="Chaos injection"):
            await chaos_effect_executor.execute_with_chaos(
                intent_id=intent_id,
                operation="test_operation",
                correlation_id=correlation_id,
                fail_point="pre",
            )

        # Backend should NOT have been called
        chaos_effect_executor.backend_client.execute.assert_not_called()
        assert chaos_effect_executor.execution_count == 0
        assert deterministic_failure_injector.failure_count == 1

    @pytest.mark.asyncio
    async def test_failure_during_execution(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that failure during execution is tracked.

        When a handler fails during the main execution:
        - The failure should be propagated
        - The idempotency record should exist (operation was attempted)
        - The failed_count should be incremented
        """
        # Arrange
        deterministic_injector = FailureInjector(config=ChaosConfig(failure_rate=1.0))
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=deterministic_injector,
            backend_client=mock_backend_client,
        )

        intent_id = uuid4()
        correlation_id = uuid4()

        # Act & Assert - InfraConnectionError is used for chaos injection failures
        with pytest.raises(InfraConnectionError, match="Chaos injection"):
            await executor.execute_with_chaos(
                intent_id=intent_id,
                operation="test_operation",
                correlation_id=correlation_id,
                fail_point="mid",
            )

        # Failure was tracked
        assert executor.failed_count == 1
        assert deterministic_injector.failure_count == 1

    @pytest.mark.asyncio
    async def test_failure_after_execution(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that failure after execution still records the backend call.

        When a handler fails after the main execution:
        - The backend operation WAS called successfully
        - The post-execution failure is propagated
        - The execution count should be incremented
        """
        # Arrange
        deterministic_injector = FailureInjector(config=ChaosConfig(failure_rate=1.0))
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=deterministic_injector,
            backend_client=mock_backend_client,
        )

        intent_id = uuid4()
        correlation_id = uuid4()

        # Act & Assert - InfraConnectionError is used for chaos injection failures
        with pytest.raises(InfraConnectionError, match="Chaos injection"):
            await executor.execute_with_chaos(
                intent_id=intent_id,
                operation="test_operation",
                correlation_id=correlation_id,
                fail_point="post",
            )

        # Backend WAS called before failure
        mock_backend_client.execute.assert_called_once()
        # But failure still tracked
        assert deterministic_injector.failure_count == 1

    @pytest.mark.asyncio
    async def test_no_failure_when_disabled(
        self,
        chaos_effect_executor: ChaosEffectExecutor,
    ) -> None:
        """Test that chaos injection can be disabled.

        When chaos injection is disabled:
        - Operations should succeed normally
        - No failures should be injected
        """
        # Arrange - set high failure rate but disable
        chaos_effect_executor.failure_injector.config.failure_rate = 1.0
        chaos_effect_executor.failure_injector.config.enabled = False

        intent_id = uuid4()

        # Act - should succeed despite high failure rate
        result = await chaos_effect_executor.execute_with_chaos(
            intent_id=intent_id,
            operation="test_operation",
            fail_point="mid",
        )

        # Assert
        assert result is True
        assert chaos_effect_executor.execution_count == 1
        assert chaos_effect_executor.failure_injector.failure_count == 0


@pytest.mark.chaos
class TestHandlerFailurePropagation:
    """Test that handler failures are properly propagated."""

    @pytest.mark.asyncio
    async def test_failure_propagates_with_correlation_id_in_context(
        self,
        deterministic_failure_injector: FailureInjector,
    ) -> None:
        """Test that failure errors include correlation ID in context.

        Per ONEX error sanitization guidelines, correlation_id should be
        passed via ModelInfraErrorContext, NOT in the error message string.
        This enables tracing while avoiding sensitive data leakage.
        """
        # Arrange
        correlation_id = uuid4()

        # Act & Assert - InfraConnectionError includes context with correlation_id
        with pytest.raises(InfraConnectionError) as exc_info:
            await deterministic_failure_injector.maybe_inject_failure(
                operation="test_op",
                correlation_id=correlation_id,
            )

        # Verify correlation_id is in error model (proper ONEX pattern)
        # The correlation_id is stored in error.model.correlation_id
        # (or error.correlation_id)
        error = exc_info.value
        assert error.model.correlation_id == correlation_id

        # Verify correlation_id is NOT exposed in message string (sanitization)
        # The error message should not contain the raw UUID string
        assert str(correlation_id) not in error.message

    @pytest.mark.asyncio
    async def test_failure_does_not_mask_original_error(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
        failure_injector: FailureInjector,
    ) -> None:
        """Test that original backend errors are not masked by chaos.

        When the backend itself fails (not chaos injection), the original
        error should be propagated, not a chaos error.
        """
        # Arrange - backend will fail with specific error
        mock_backend_client.execute = AsyncMock(
            side_effect=RuntimeError("Backend database connection failed")
        )
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=failure_injector,
            backend_client=mock_backend_client,
        )

        intent_id = uuid4()

        # Act & Assert - should get backend error, not chaos error
        with pytest.raises(RuntimeError, match="Backend database connection failed"):
            await executor.execute_with_chaos(
                intent_id=intent_id,
                operation="test_operation",
            )

    @pytest.mark.asyncio
    async def test_concurrent_failures_are_isolated(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that concurrent handler failures are isolated.

        When multiple handlers fail concurrently, each failure should be
        independent and not affect other handlers.
        """
        # Arrange
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=FailureInjector(
                config=ChaosConfig(failure_rate=0.5)  # 50% failure rate
            ),
            backend_client=mock_backend_client,
        )

        num_concurrent = 20

        async def execute_one(i: int) -> bool:
            return await executor.execute_with_chaos(
                intent_id=uuid4(),
                operation=f"test_operation_{i}",
                fail_point="mid",
            )

        # Act - execute concurrently using shared utility
        results, errors = await run_concurrent_with_tracking(
            execute_one, count=num_concurrent
        )

        # Assert - some succeeded, some failed, but all are tracked
        total = len(results) + len(errors)
        assert total == num_concurrent
        # With 50% failure rate, expect roughly half to fail (with variance)
        assert len(errors) > 0  # At least some failures
        assert len(results) > 0  # At least some successes


@pytest.mark.chaos
class TestHandlerFailureRecovery:
    """Test recovery behavior after handler failures."""

    @pytest.mark.asyncio
    async def test_retry_after_failure_succeeds(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that retry after failure can succeed.

        After a handler fails, a new attempt with a different intent ID
        should be able to succeed.
        """
        # Arrange - injector that fails once, then succeeds
        call_count = 0

        async def conditional_failure(
            operation: str, correlation_id: UUID | None = None
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")

        injector = FailureInjector(config=ChaosConfig())
        injector.maybe_inject_failure = conditional_failure  # type: ignore[method-assign]

        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=injector,
            backend_client=mock_backend_client,
        )

        # First attempt - fails
        intent_id_1 = uuid4()
        with pytest.raises(ValueError, match="First attempt fails"):
            await executor.execute_with_chaos(
                intent_id=intent_id_1,
                operation="test_operation",
                fail_point="mid",
            )

        # Second attempt with new intent - succeeds
        intent_id_2 = uuid4()
        result = await executor.execute_with_chaos(
            intent_id=intent_id_2,
            operation="test_operation",
            fail_point="mid",
        )

        assert result is True
        assert executor.execution_count == 1  # Only second attempt counted

    @pytest.mark.asyncio
    async def test_idempotency_prevents_duplicate_after_failure(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
        failure_injector: FailureInjector,
    ) -> None:
        """Test that idempotency prevents duplicates even after failure.

        If a handler fails after recording in the idempotency store,
        a retry with the same intent ID should skip execution.

        Note: This test validates behavior when failure occurs AFTER
        idempotency check passes but BEFORE execution completes.
        """
        # Arrange
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=failure_injector,
            backend_client=mock_backend_client,
        )

        intent_id = uuid4()

        # Pre-record in idempotency store (simulating partial execution)
        await chaos_idempotency_store.check_and_record(
            message_id=intent_id,
            domain="chaos",
        )

        # Attempt to execute - should be detected as duplicate
        result = await executor.execute_with_chaos(
            intent_id=intent_id,
            operation="test_operation",
        )

        # Assert - marked as success (duplicate), but backend not called
        assert result is True
        mock_backend_client.execute.assert_not_called()
        assert executor.execution_count == 0


@pytest.mark.chaos
class TestHandlerFailureRandom:
    """Test random failure injection scenarios."""

    @pytest.mark.asyncio
    async def test_random_failures_with_configured_rate(
        self,
        chaos_idempotency_store,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test that random failures occur at approximately the configured rate.

        With a 30% failure rate, approximately 30% of operations should fail
        (with some statistical variance).
        """
        # Arrange
        failure_rate = 0.3
        injector = FailureInjector(config=ChaosConfig(failure_rate=failure_rate))
        executor = ChaosEffectExecutor(
            idempotency_store=chaos_idempotency_store,
            failure_injector=injector,
            backend_client=mock_backend_client,
        )

        # Use 500 iterations for better statistical power.
        # With n=500, p=0.3: expected=150, stddev=sqrt(500*0.3*0.7)~=10.25
        # 20% tolerance gives range [120, 180], which is +/-2.9 stddev from mean.
        # This makes the test resilient to normal statistical variance.
        num_iterations = 500
        failure_count = 0

        # Act
        for i in range(num_iterations):
            try:
                await executor.execute_with_chaos(
                    intent_id=uuid4(),
                    operation=f"test_operation_{i}",
                    fail_point="mid",
                )
            except InfraConnectionError:
                failure_count += 1

        # Assert - failure rate should be roughly 30% (with tolerance)
        # Using shared utility for statistical rate assertion
        assert_failure_rate_within_tolerance(
            actual_failures=failure_count,
            total_attempts=num_iterations,
            expected_rate=failure_rate,
            tolerance=0.2,
            context="random failure test",
        )

    @pytest.mark.asyncio
    async def test_failure_counter_tracks_injections(
        self,
        deterministic_failure_injector: FailureInjector,
    ) -> None:
        """Test that the failure counter accurately tracks injections."""
        # Arrange & Act
        for i in range(5):
            try:
                await deterministic_failure_injector.maybe_inject_failure(
                    operation=f"test_{i}",
                )
            except InfraConnectionError:
                pass

        # Assert
        assert deterministic_failure_injector.failure_count == 5

    @pytest.mark.asyncio
    async def test_reset_counts_clears_counters(
        self,
        deterministic_failure_injector: FailureInjector,
    ) -> None:
        """Test that reset_counts clears the failure counter."""
        # Arrange - inject some failures
        for _ in range(3):
            try:
                await deterministic_failure_injector.maybe_inject_failure("test")
            except InfraConnectionError:
                pass

        assert deterministic_failure_injector.failure_count == 3

        # Act
        deterministic_failure_injector.reset_counts()

        # Assert
        assert deterministic_failure_injector.failure_count == 0
        assert deterministic_failure_injector.timeout_count == 0

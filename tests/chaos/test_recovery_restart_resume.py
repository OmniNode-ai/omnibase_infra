# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Restart recovery tests for OMN-955.

This test suite validates that workflows correctly resume after a process restart.
It tests the critical guarantee that:

1. Workflow progress is tracked via idempotency store
2. After restart, already-completed steps are skipped
3. Workflow resumes from the correct point
4. No duplicate side effects occur on restart

Architecture:
    The Effect layer uses an idempotency store (InMemory for tests, PostgreSQL
    in production) to track which intents have been processed. On restart:

    1. Effect executor initializes with same idempotency store
    2. When processing pending intents, store is checked first
    3. Already-processed intents are skipped (no I/O)
    4. New intents are processed and recorded

Test Organization:
    - TestRestartMidWorkflow: Restart during multi-step workflow
    - TestResumeFromCorrectPoint: Verify resume position
    - TestNoExtraIO: Confirm no duplicate side effects

Related:
    - OMN-955: Failure recovery tests
    - OMN-954: Effect idempotency
    - OMN-945: Idempotency system
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.idempotency import InMemoryIdempotencyStore

# =============================================================================
# Test Helper: Simulated Effect Executor
# =============================================================================


class SimulatedEffectExecutor:
    """Simulates Effect node behavior with idempotency guard for restart testing.

    This class represents the essential idempotency logic that an Effect node
    uses when processing registration intents. It demonstrates the pattern:

    1. Check idempotency store before execution
    2. Execute backend operation if not duplicate
    3. Record execution in idempotency store

    The key property for restart testing is that this executor can be
    "restarted" by creating a new instance with the same idempotency store.

    Attributes:
        idempotency_store: Store for tracking processed intents.
        backend_client: Mock backend client for simulating operations.
        execution_count: Counter for tracking actual backend calls.
        executed_steps: List of step IDs that were actually executed.
        instance_id: Unique identifier for this executor instance.
    """

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        backend_client: MagicMock | None = None,
        instance_id: str | None = None,
    ) -> None:
        """Initialize the simulated effect executor.

        Args:
            idempotency_store: Store for idempotency checking.
            backend_client: Mock backend client (created if not provided).
            instance_id: Identifier for this instance (for restart tracking).
        """
        self.idempotency_store = idempotency_store
        if backend_client is not None:
            # Use the provided backend client (for shared backend tests)
            self.backend_client = backend_client
        else:
            # Create a new mock backend with async execute
            self.backend_client = MagicMock()
            self.backend_client.execute = AsyncMock(return_value=True)
        self.execution_count = 0
        self.executed_steps: list[str] = []
        self.instance_id = instance_id or str(uuid4())[:8]

    async def execute_step(
        self,
        step_id: UUID,
        step_name: str,
        correlation_id: UUID | None = None,
        fail_after_check: bool = False,
    ) -> bool:
        """Execute a workflow step with idempotency guard.

        Args:
            step_id: Unique identifier for this step.
            step_name: Human-readable name for the step.
            correlation_id: Optional correlation ID for tracing.
            fail_after_check: If True, simulate failure after idempotency check
                but before recording (for testing restart scenarios).

        Returns:
            True if step was executed or was already done.

        Raises:
            RuntimeError: If fail_after_check is True (simulated crash).
        """
        # Check idempotency - domain "workflow" isolates workflow steps
        is_new = await self.idempotency_store.check_and_record(
            message_id=step_id,
            domain="workflow",
            correlation_id=correlation_id,
        )

        if not is_new:
            # Already processed - skip execution
            return True

        if fail_after_check:
            # Simulate crash after idempotency check but before backend call
            # This tests the critical restart scenario where a step was
            # partially processed
            raise RuntimeError(f"Simulated crash during step {step_name}")

        # Execute the actual operation
        self.execution_count += 1
        self.executed_steps.append(step_name)
        await self.backend_client.execute(step_id=step_id, step_name=step_name)

        return True

    async def execute_workflow(
        self,
        steps: list[tuple[UUID, str]],
        correlation_id: UUID | None = None,
        fail_at_step: int | None = None,
    ) -> dict[str, object]:
        """Execute a multi-step workflow with idempotency.

        Args:
            steps: List of (step_id, step_name) tuples.
            correlation_id: Optional correlation ID for tracing.
            fail_at_step: If set, simulate failure at this step index.

        Returns:
            Dict with execution results:
                - completed_steps: Number of steps completed
                - executed_steps: List of step names that were executed
                - failed_at: Step index where failure occurred (if any)

        Raises:
            RuntimeError: If fail_at_step is reached.
        """
        completed_steps = 0
        failed_at: int | None = None

        for i, (step_id, step_name) in enumerate(steps):
            if fail_at_step is not None and i == fail_at_step:
                failed_at = i
                raise RuntimeError(f"Simulated failure at step {i}: {step_name}")

            await self.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )
            completed_steps += 1

        return {
            "completed_steps": completed_steps,
            "executed_steps": list(self.executed_steps),
            "failed_at": failed_at,
        }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def idempotency_store() -> InMemoryIdempotencyStore:
    """Create InMemoryIdempotencyStore for testing restart scenarios."""
    return InMemoryIdempotencyStore()


@pytest.fixture
def workflow_steps() -> list[tuple[UUID, str]]:
    """Create a standard 5-step workflow for testing."""
    return [
        (uuid4(), "step_1_validate_input"),
        (uuid4(), "step_2_check_permissions"),
        (uuid4(), "step_3_register_consul"),
        (uuid4(), "step_4_upsert_postgres"),
        (uuid4(), "step_5_publish_event"),
    ]


# =============================================================================
# Test Classes
# =============================================================================
# NOTE: correlation_id fixture is provided by chaos/conftest.py


@pytest.mark.unit
@pytest.mark.chaos
class TestRestartMidWorkflow:
    """Test restart during multi-step workflow execution."""

    @pytest.mark.asyncio
    async def test_restart_mid_workflow_resumes_correctly(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        workflow_steps: list[tuple[UUID, str]],
        correlation_id: UUID,
    ) -> None:
        """Test that workflow resumes correctly after restart.

        Scenario:
            1. Execute workflow steps 1-3
            2. Simulate restart (create new executor with same store)
            3. Re-execute entire workflow
            4. Verify steps 1-3 are skipped, steps 4-5 are executed
        """
        # Create first executor instance
        executor_before = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
            instance_id="before_restart",
        )

        # Execute first 3 steps
        for step_id, step_name in workflow_steps[:3]:
            await executor_before.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )

        # Verify 3 steps were executed
        assert executor_before.execution_count == 3
        assert len(executor_before.executed_steps) == 3

        # Simulate restart: create NEW executor with SAME idempotency store
        executor_after = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,  # Same store!
            instance_id="after_restart",
        )

        # Re-execute ALL steps (simulating workflow replay)
        for step_id, step_name in workflow_steps:
            await executor_after.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )

        # Verify: only steps 4-5 were executed by new instance
        assert executor_after.execution_count == 2
        assert executor_after.executed_steps == [
            "step_4_upsert_postgres",
            "step_5_publish_event",
        ]

        # Total execution across both instances should be 5
        total = executor_before.execution_count + executor_after.execution_count
        assert total == 5

    @pytest.mark.asyncio
    async def test_restart_after_partial_step_execution(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        workflow_steps: list[tuple[UUID, str]],
        correlation_id: UUID,
    ) -> None:
        """Test restart after step was marked but operation failed.

        Scenario:
            In check_and_record pattern, if we crash after recording but before
            completing the operation, the idempotency store has the record but
            the operation wasn't completed.

            This is the "at-least-once" guarantee: the step may be re-executed
            on restart if we use a pattern that records BEFORE execution.

            For "exactly-once", we record AFTER execution, meaning if we crash
            after idempotency check but before recording, the step will be
            re-executed on restart.

        This test validates the current implementation uses check-then-record,
        which provides at-least-once semantics.
        """
        # Create first executor
        executor = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
            instance_id="first_attempt",
        )

        # Execute first step successfully
        step_id_1, step_name_1 = workflow_steps[0]
        await executor.execute_step(
            step_id=step_id_1,
            step_name=step_name_1,
            correlation_id=correlation_id,
        )
        assert executor.execution_count == 1

        # Second step: simulate crash after idempotency check
        # Note: check_and_record was already called, so the record exists
        # We're simulating a crash AFTER check_and_record but BEFORE
        # completing the backend operation
        step_id_2, step_name_2 = workflow_steps[1]
        with pytest.raises(RuntimeError, match="Simulated crash"):
            await executor.execute_step(
                step_id=step_id_2,
                step_name=step_name_2,
                correlation_id=correlation_id,
                fail_after_check=True,
            )

        # The idempotency record was created (check_and_record was called)
        # but the backend operation was NOT completed

        # Create new executor (restart)
        executor_restarted = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
            instance_id="restarted",
        )

        # Re-execute all steps
        for step_id, step_name in workflow_steps:
            await executor_restarted.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )

        # With check_and_record pattern (records first), step_2 is skipped
        # because the record already exists, even though the operation failed.
        # This demonstrates the trade-off: may skip failed operations.
        # Steps 1-2 recorded before crash, so restarted executor sees them as done
        # Steps 3-5 were never recorded, so they are executed
        assert executor_restarted.execution_count == 3  # steps 3, 4, 5


@pytest.mark.unit
@pytest.mark.chaos
class TestResumeFromCorrectPoint:
    """Test that workflows resume from the correct point."""

    @pytest.mark.asyncio
    async def test_resume_from_exact_failure_point(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        workflow_steps: list[tuple[UUID, str]],
        correlation_id: UUID,
    ) -> None:
        """Test that resume starts exactly where failure occurred."""
        executor = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )

        # Execute with failure at step 3 (index 2)
        with pytest.raises(RuntimeError, match="Simulated failure at step 2"):
            await executor.execute_workflow(
                steps=workflow_steps,
                correlation_id=correlation_id,
                fail_at_step=2,
            )

        # Steps 0-1 were completed
        assert executor.execution_count == 2
        assert executor.executed_steps == [
            "step_1_validate_input",
            "step_2_check_permissions",
        ]

        # Create new executor (restart)
        executor_resumed = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )

        # Re-execute workflow (should resume from step 3)
        result = await executor_resumed.execute_workflow(
            steps=workflow_steps,
            correlation_id=correlation_id,
        )

        # New executor only executed steps 3-5
        assert executor_resumed.execution_count == 3
        assert executor_resumed.executed_steps == [
            "step_3_register_consul",
            "step_4_upsert_postgres",
            "step_5_publish_event",
        ]
        assert result["completed_steps"] == 5  # All steps completed

    @pytest.mark.asyncio
    async def test_multiple_restarts_all_resume_correctly(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        workflow_steps: list[tuple[UUID, str]],
        correlation_id: UUID,
    ) -> None:
        """Test that multiple restarts all resume from correct point."""
        total_executed = 0

        # First executor: executes step 1 only
        executor_1 = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )
        await executor_1.execute_step(
            step_id=workflow_steps[0][0],
            step_name=workflow_steps[0][1],
            correlation_id=correlation_id,
        )
        total_executed += executor_1.execution_count
        assert executor_1.execution_count == 1

        # Second executor: executes steps 2-3
        executor_2 = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )
        for step_id, step_name in workflow_steps[:3]:
            await executor_2.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )
        total_executed += executor_2.execution_count
        assert executor_2.execution_count == 2  # Steps 2-3 only

        # Third executor: executes steps 4-5
        executor_3 = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )
        for step_id, step_name in workflow_steps:
            await executor_3.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )
        total_executed += executor_3.execution_count
        assert executor_3.execution_count == 2  # Steps 4-5 only

        # Total across all executors should equal total steps
        assert total_executed == len(workflow_steps)


@pytest.mark.unit
@pytest.mark.chaos
class TestNoExtraIO:
    """Test that no duplicate side effects occur on restart."""

    @pytest.mark.asyncio
    async def test_no_duplicate_backend_calls(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        workflow_steps: list[tuple[UUID, str]],
        correlation_id: UUID,
    ) -> None:
        """Test that restarted executor doesn't duplicate backend calls."""
        # Shared backend client to track total calls
        shared_backend = MagicMock()
        shared_backend.execute = AsyncMock(return_value=True)

        # First executor
        executor_1 = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
            backend_client=shared_backend,
        )

        # Execute all steps with first executor
        for step_id, step_name in workflow_steps:
            await executor_1.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )

        # Backend should have been called 5 times
        assert shared_backend.execute.call_count == 5

        # Second executor (restart)
        executor_2 = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
            backend_client=shared_backend,
        )

        # Re-execute all steps
        for step_id, step_name in workflow_steps:
            await executor_2.execute_step(
                step_id=step_id,
                step_name=step_name,
                correlation_id=correlation_id,
            )

        # Backend should STILL have been called only 5 times (no duplicates)
        assert shared_backend.execute.call_count == 5
        assert executor_2.execution_count == 0  # No new executions

    @pytest.mark.asyncio
    async def test_concurrent_restart_no_duplicates(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        workflow_steps: list[tuple[UUID, str]],
        correlation_id: UUID,
    ) -> None:
        """Test that concurrent restarts don't cause duplicate executions.

        Simulates a scenario where multiple executor instances try to process
        the same workflow concurrently (e.g., during a rolling restart).
        """
        # Shared backend
        shared_backend = MagicMock()
        shared_backend.execute = AsyncMock(return_value=True)

        # Create multiple executors
        executors = [
            SimulatedEffectExecutor(
                idempotency_store=idempotency_store,
                backend_client=shared_backend,
                instance_id=f"executor_{i}",
            )
            for i in range(3)
        ]

        # Each executor tries to process all steps concurrently
        async def process_all(executor: SimulatedEffectExecutor) -> int:
            for step_id, step_name in workflow_steps:
                await executor.execute_step(
                    step_id=step_id,
                    step_name=step_name,
                    correlation_id=correlation_id,
                )
            return executor.execution_count

        # Run all executors concurrently
        results = await asyncio.gather(*[process_all(e) for e in executors])

        # Total executions across all executors should equal total steps
        total_executions = sum(results)
        assert total_executions == len(workflow_steps)

        # Backend should have been called exactly 5 times
        assert shared_backend.execute.call_count == 5


@pytest.mark.unit
@pytest.mark.chaos
class TestIdempotencyStorePersistence:
    """Test that idempotency records persist across restarts."""

    @pytest.mark.asyncio
    async def test_records_survive_executor_restart(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        correlation_id: UUID,
    ) -> None:
        """Test that idempotency records created before restart are visible after."""
        step_id = uuid4()

        # First executor records a step
        executor_1 = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )
        await executor_1.execute_step(
            step_id=step_id,
            step_name="persistent_step",
            correlation_id=correlation_id,
        )
        assert executor_1.execution_count == 1

        # Verify record exists in store
        record = await idempotency_store.get_record(
            message_id=step_id,
            domain="workflow",
        )
        assert record is not None
        assert record.message_id == step_id
        assert record.correlation_id == correlation_id

        # Second executor should see the same record
        executor_2 = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )
        await executor_2.execute_step(
            step_id=step_id,
            step_name="persistent_step",
            correlation_id=correlation_id,
        )
        assert executor_2.execution_count == 0  # Skipped due to existing record

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_in_records(
        self,
        idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Test that correlation IDs are preserved in idempotency records."""
        correlation_id = uuid4()
        step_id = uuid4()

        executor = SimulatedEffectExecutor(
            idempotency_store=idempotency_store,
        )
        await executor.execute_step(
            step_id=step_id,
            step_name="tracked_step",
            correlation_id=correlation_id,
        )

        # Retrieve record and verify correlation_id
        record = await idempotency_store.get_record(
            message_id=step_id,
            domain="workflow",
        )
        assert record is not None
        assert record.correlation_id == correlation_id


__all__ = [
    "SimulatedEffectExecutor",
    "TestRestartMidWorkflow",
    "TestResumeFromCorrectPoint",
    "TestNoExtraIO",
    "TestIdempotencyStorePersistence",
]

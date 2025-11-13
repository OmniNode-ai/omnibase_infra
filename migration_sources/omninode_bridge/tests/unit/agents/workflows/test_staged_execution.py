"""
Tests for staged parallel execution system.

This module tests the 6-phase code generation pipeline with parallel
contract processing.

Performance targets:
- Full pipeline: <5s for typical contract
- Stage transition overhead: <100ms
- Parallelism speedup: 2.25x-4.17x
"""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from omninode_bridge.agents.coordination.signals import SignalCoordinator
from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.scheduler.scheduler import DependencyAwareScheduler
from omninode_bridge.agents.workflows.staged_execution import StagedParallelExecutor
from omninode_bridge.agents.workflows.workflow_models import (
    EnumStageStatus,
    EnumStepType,
    WorkflowConfig,
    WorkflowStage,
    WorkflowStep,
)


class TestWorkflowModels:
    """Test workflow model validation and properties."""

    def test_workflow_step_creation(self):
        """Test WorkflowStep creation and validation."""
        async def dummy_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {"result": "success"}

        step = WorkflowStep(
            step_id="parse_contract_1",
            step_type=EnumStepType.PARSE_CONTRACT,
            dependencies=[],
            input_data={"contract_path": "test.yaml"},
            executor=dummy_executor,
        )

        assert step.step_id == "parse_contract_1"
        assert step.step_type == EnumStepType.PARSE_CONTRACT
        assert step.dependencies == []
        assert step.status == EnumStageStatus.PENDING
        assert step.result is None
        assert step.error is None

    def test_workflow_step_validation_empty_id(self):
        """Test WorkflowStep validation with empty step_id."""
        async def dummy_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {}

        with pytest.raises(ValueError, match="step_id cannot be empty"):
            WorkflowStep(
                step_id="",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={},
                executor=dummy_executor,
            )

    def test_workflow_step_validation_non_callable_executor(self):
        """Test WorkflowStep validation with non-callable executor."""
        with pytest.raises(ValueError, match="executor must be callable"):
            WorkflowStep(
                step_id="test",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={},
                executor="not_callable",  # type: ignore
            )

    def test_workflow_stage_creation(self):
        """Test WorkflowStage creation and validation."""
        async def dummy_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {}

        steps = [
            WorkflowStep(
                step_id=f"step_{i}",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={},
                executor=dummy_executor,
            )
            for i in range(3)
        ]

        stage = WorkflowStage(
            stage_id="parse",
            stage_number=1,
            stage_name="Parse Contracts",
            steps=steps,
            dependencies=[],
            parallel=True,
        )

        assert stage.stage_id == "parse"
        assert stage.stage_number == 1
        assert stage.total_steps == 3
        assert stage.completed_steps == 0
        assert stage.failed_steps == 0
        assert not stage.is_complete
        assert not stage.has_failures

    def test_workflow_stage_validation_empty_id(self):
        """Test WorkflowStage validation with empty stage_id."""
        async def dummy_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {}

        steps = [
            WorkflowStep(
                step_id="step_1",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={},
                executor=dummy_executor,
            )
        ]

        with pytest.raises(ValueError, match="stage_id cannot be empty"):
            WorkflowStage(
                stage_id="",
                stage_number=1,
                stage_name="Test",
                steps=steps,
            )

    def test_workflow_stage_validation_invalid_stage_number(self):
        """Test WorkflowStage validation with invalid stage_number."""
        async def dummy_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {}

        steps = [
            WorkflowStep(
                step_id="step_1",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={},
                executor=dummy_executor,
            )
        ]

        with pytest.raises(ValueError, match="stage_number must be >= 1"):
            WorkflowStage(
                stage_id="test",
                stage_number=0,
                stage_name="Test",
                steps=steps,
            )

    def test_workflow_stage_validation_no_steps(self):
        """Test WorkflowStage validation with empty steps list."""
        with pytest.raises(ValueError, match="must have at least one step"):
            WorkflowStage(
                stage_id="test",
                stage_number=1,
                stage_name="Test",
                steps=[],
            )

    def test_workflow_config_validation(self):
        """Test WorkflowConfig validation."""
        config = WorkflowConfig(
            workflow_id="test-workflow",
            workflow_name="Test Workflow",
            max_concurrent_steps=10,
        )

        assert config.workflow_id == "test-workflow"
        assert config.max_concurrent_steps == 10
        assert config.enable_stage_recovery is True
        assert config.enable_step_retry is True
        assert config.step_retry_count == 2

    def test_workflow_config_validation_empty_id(self):
        """Test WorkflowConfig validation with empty workflow_id."""
        with pytest.raises(ValueError, match="workflow_id cannot be empty"):
            WorkflowConfig(
                workflow_id="",
                workflow_name="Test",
            )

    def test_workflow_config_validation_invalid_max_concurrent(self):
        """Test WorkflowConfig validation with invalid max_concurrent_steps."""
        with pytest.raises(ValueError, match="max_concurrent_steps must be >= 1"):
            WorkflowConfig(
                workflow_id="test",
                workflow_name="Test",
                max_concurrent_steps=0,
            )


class TestStagedParallelExecutor:
    """Test staged parallel executor functionality."""

    @pytest.fixture
    def state(self):
        """Create ThreadSafeState for testing."""
        return ThreadSafeState()

    @pytest.fixture
    def scheduler(self, state):
        """Create DependencyAwareScheduler for testing."""
        return DependencyAwareScheduler(state=state, max_concurrent=10)

    @pytest.fixture
    def metrics(self):
        """Create mock MetricsCollector for testing."""
        metrics = AsyncMock(spec=MetricsCollector)
        metrics.record_timing = AsyncMock()
        metrics.record_counter = AsyncMock()
        metrics.record_gauge = AsyncMock()
        return metrics

    @pytest.fixture
    def signals(self, state):
        """Create mock SignalCoordinator for testing."""
        signals = AsyncMock(spec=SignalCoordinator)
        signals.signal_coordination_event = AsyncMock(return_value=True)
        return signals

    @pytest.fixture
    def executor(self, scheduler, metrics, signals, state):
        """Create StagedParallelExecutor for testing."""
        return StagedParallelExecutor(
            scheduler=scheduler,
            metrics_collector=metrics,
            signal_coordinator=signals,
            state=state,
        )

    @pytest.mark.asyncio
    async def test_single_stage_execution(self, executor):
        """Test execution of a single stage with parallel steps."""
        call_count = 0

        async def test_executor(context: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return {"task_id": context.get("task_id", "unknown"), "result": "success"}

        # Create stage with 3 parallel steps
        steps = [
            WorkflowStep(
                step_id=f"step_{i}",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={"contract_id": i},
                executor=test_executor,
            )
            for i in range(3)
        ]

        stage = WorkflowStage(
            stage_id="parse",
            stage_number=1,
            stage_name="Parse Contracts",
            steps=steps,
            parallel=True,
        )

        # Execute workflow
        result = await executor.execute_workflow(
            workflow_id="test-workflow-1",
            stages=[stage],
        )

        # Verify results
        assert result.status == EnumStageStatus.COMPLETED
        assert result.total_stages == 1
        assert result.successful_stages == 1
        assert result.failed_stages == 0
        assert result.total_steps == 3
        assert result.successful_steps == 3
        assert result.failed_steps == 0
        assert call_count == 3  # All steps executed

        # Verify speedup (should be >1 for parallel execution)
        assert result.overall_speedup > 1.0

    @pytest.mark.asyncio
    async def test_multi_stage_sequential_execution(self, executor):
        """Test execution of multiple stages in sequence."""
        execution_order = []

        async def stage1_executor(context: dict[str, Any]) -> dict[str, Any]:
            task_id = context.get("task_id", "unknown")
            execution_order.append(f"stage1_{task_id}")
            await asyncio.sleep(0.01)
            return {"stage": 1, "task_id": task_id}

        async def stage2_executor(context: dict[str, Any]) -> dict[str, Any]:
            task_id = context.get("task_id", "unknown")
            execution_order.append(f"stage2_{task_id}")
            await asyncio.sleep(0.01)
            return {"stage": 2, "task_id": task_id}

        # Stage 1: Parse
        stage1 = WorkflowStage(
            stage_id="parse",
            stage_number=1,
            stage_name="Parse",
            steps=[
                WorkflowStep(
                    step_id="parse_1",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=stage1_executor,
                )
            ],
            parallel=True,
        )

        # Stage 2: Generate (depends on stage 1)
        stage2 = WorkflowStage(
            stage_id="generate",
            stage_number=2,
            stage_name="Generate",
            steps=[
                WorkflowStep(
                    step_id="generate_1",
                    step_type=EnumStepType.GENERATE_MODEL,
                    dependencies=[],
                    input_data={},
                    executor=stage2_executor,
                )
            ],
            dependencies=["parse"],
            parallel=True,
        )

        # Execute workflow
        result = await executor.execute_workflow(
            workflow_id="test-workflow-2",
            stages=[stage1, stage2],
        )

        # Verify results
        assert result.status == EnumStageStatus.COMPLETED
        assert result.total_stages == 2
        assert result.successful_stages == 2
        assert result.total_steps == 2
        assert result.successful_steps == 2

        # Verify execution order (stage 1 before stage 2)
        assert execution_order.index("stage1_parse_1") < execution_order.index(
            "stage2_generate_1"
        )

    @pytest.mark.asyncio
    async def test_step_dependencies_within_stage(self, executor):
        """Test step dependencies within a single stage."""
        execution_order = []

        async def step_executor(context: dict[str, Any]) -> dict[str, Any]:
            task_id = context.get("task_id", "unknown")
            execution_order.append(task_id)
            await asyncio.sleep(0.01)
            return {"task_id": task_id}

        # Create steps with dependencies: step1 → step2 → step3
        steps = [
            WorkflowStep(
                step_id="step1",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={},
                executor=step_executor,
            ),
            WorkflowStep(
                step_id="step2",
                step_type=EnumStepType.GENERATE_MODEL,
                dependencies=["step1"],
                input_data={},
                executor=step_executor,
            ),
            WorkflowStep(
                step_id="step3",
                step_type=EnumStepType.GENERATE_VALIDATOR,
                dependencies=["step2"],
                input_data={},
                executor=step_executor,
            ),
        ]

        stage = WorkflowStage(
            stage_id="codegen",
            stage_number=1,
            stage_name="Code Generation",
            steps=steps,
            parallel=True,  # Parallel execution with dependencies
        )

        # Execute workflow
        result = await executor.execute_workflow(
            workflow_id="test-workflow-3",
            stages=[stage],
        )

        # Verify results
        assert result.status == EnumStageStatus.COMPLETED
        assert result.successful_steps == 3

        # Verify execution order respects dependencies
        assert execution_order.index("step1") < execution_order.index("step2")
        assert execution_order.index("step2") < execution_order.index("step3")

    @pytest.mark.asyncio
    async def test_parallel_speedup_measurement(self, executor):
        """Test parallel speedup measurement."""

        async def slow_executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate slow work
            return {"task_id": context.get("task_id", "unknown")}

        # Create stage with 5 independent parallel steps
        steps = [
            WorkflowStep(
                step_id=f"step_{i}",
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={},
                executor=slow_executor,
            )
            for i in range(5)
        ]

        stage = WorkflowStage(
            stage_id="parse",
            stage_number=1,
            stage_name="Parse",
            steps=steps,
            parallel=True,
        )

        # Execute workflow
        result = await executor.execute_workflow(
            workflow_id="test-workflow-4",
            stages=[stage],
        )

        # Verify speedup
        # With 5 parallel steps of 100ms each:
        # - Sequential time: ~500ms
        # - Parallel time: ~100ms (+ overhead)
        # - Expected speedup: ~3-5x
        assert result.status == EnumStageStatus.COMPLETED
        assert result.successful_steps == 5
        assert result.overall_speedup >= 2.0  # Conservative estimate
        print(f"Actual speedup: {result.overall_speedup:.2f}x")

    @pytest.mark.asyncio
    async def test_stage_dependency_blocking(self, executor):
        """Test that stage dependencies properly block execution."""

        async def executor_fn(context: dict[str, Any]) -> dict[str, Any]:
            return {"task_id": context.get("task_id", "unknown")}

        # Stage 1: Success
        stage1 = WorkflowStage(
            stage_id="stage1",
            stage_number=1,
            stage_name="Stage 1",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=executor_fn,
                )
            ],
        )

        # Stage 2: Depends on stage1 (will be skipped because stage1 fails)
        stage2 = WorkflowStage(
            stage_id="stage2",
            stage_number=2,
            stage_name="Stage 2",
            steps=[
                WorkflowStep(
                    step_id="step2",
                    step_type=EnumStepType.GENERATE_MODEL,
                    dependencies=[],
                    input_data={},
                    executor=executor_fn,
                )
            ],
            dependencies=["stage1"],
        )

        # Make stage1 fail
        async def failing_executor(context: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("Intentional failure")

        stage1.steps[0].executor = failing_executor

        # Execute workflow
        result = await executor.execute_workflow(
            workflow_id="test-workflow-5",
            stages=[stage1, stage2],
        )

        # Verify results
        assert result.status == EnumStageStatus.FAILED
        assert result.failed_stages >= 1
        # Stage 2 should be skipped due to stage1 failure
        stage2_result = next((sr for sr in result.stage_results if sr.stage_id == "stage2"), None)
        assert stage2_result is not None
        assert stage2_result.status == EnumStageStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_error_handling_with_recovery(self, executor):
        """Test error handling with stage recovery enabled."""
        success_count = 0
        fail_count = 0

        async def success_executor(context: dict[str, Any]) -> dict[str, Any]:
            nonlocal success_count
            success_count += 1
            return {"result": "success"}

        async def fail_executor(context: dict[str, Any]) -> dict[str, Any]:
            nonlocal fail_count
            fail_count += 1
            raise ValueError("Step failed")

        # Stage 1: Will fail
        stage1 = WorkflowStage(
            stage_id="failing_stage",
            stage_number=1,
            stage_name="Failing Stage",
            steps=[
                WorkflowStep(
                    step_id="fail_step",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=fail_executor,
                )
            ],
        )

        # Stage 2: Will succeed (recovery enabled)
        stage2 = WorkflowStage(
            stage_id="success_stage",
            stage_number=2,
            stage_name="Success Stage",
            steps=[
                WorkflowStep(
                    step_id="success_step",
                    step_type=EnumStepType.GENERATE_MODEL,
                    dependencies=[],
                    input_data={},
                    executor=success_executor,
                )
            ],
        )

        # Execute with recovery enabled (and retries disabled for predictable test)
        config = WorkflowConfig(
            workflow_id="test-workflow-6",
            workflow_name="Recovery Test",
            enable_stage_recovery=True,
            enable_step_retry=False,  # Disable retries for predictable count
            step_retry_count=0,
        )

        result = await executor.execute_workflow(
            workflow_id="test-workflow-6",
            stages=[stage1, stage2],
            config=config,
        )

        # Verify results
        assert result.status == EnumStageStatus.FAILED  # Overall failed
        assert result.failed_stages == 1  # Stage 1 failed
        assert result.successful_stages == 1  # Stage 2 succeeded (recovery)
        assert fail_count == 1  # Only called once (no retries)
        assert success_count == 1

    @pytest.mark.asyncio
    async def test_error_handling_without_recovery(self, executor):
        """Test error handling with stage recovery disabled."""
        call_count = 0

        async def count_executor(context: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First stage fails")
            return {"result": "success"}

        # Stage 1: Will fail
        stage1 = WorkflowStage(
            stage_id="failing_stage",
            stage_number=1,
            stage_name="Failing Stage",
            steps=[
                WorkflowStep(
                    step_id="fail_step",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=count_executor,
                )
            ],
        )

        # Stage 2: Should not execute (recovery disabled)
        stage2 = WorkflowStage(
            stage_id="skipped_stage",
            stage_number=2,
            stage_name="Skipped Stage",
            steps=[
                WorkflowStep(
                    step_id="skipped_step",
                    step_type=EnumStepType.GENERATE_MODEL,
                    dependencies=[],
                    input_data={},
                    executor=count_executor,
                )
            ],
        )

        # Execute with recovery disabled (and retries disabled for predictable test)
        config = WorkflowConfig(
            workflow_id="test-workflow-7",
            workflow_name="No Recovery Test",
            enable_stage_recovery=False,
            enable_step_retry=False,  # Disable retries for predictable count
            step_retry_count=0,
        )

        result = await executor.execute_workflow(
            workflow_id="test-workflow-7",
            stages=[stage1, stage2],
            config=config,
        )

        # Verify results
        assert result.status == EnumStageStatus.FAILED
        assert result.failed_stages == 1
        assert call_count == 1  # Only stage 1 executed (no retries)

    @pytest.mark.asyncio
    async def test_metrics_collection(self, executor, metrics):
        """Test that metrics are collected during workflow execution."""

        async def test_executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"result": "success"}

        stage = WorkflowStage(
            stage_id="test_stage",
            stage_number=1,
            stage_name="Test Stage",
            steps=[
                WorkflowStep(
                    step_id="test_step",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=test_executor,
                )
            ],
        )

        # Execute workflow
        await executor.execute_workflow(
            workflow_id="test-workflow-8",
            stages=[stage],
        )

        # Verify metrics were collected
        metrics.record_timing.assert_called()
        metrics.record_gauge.assert_called()

        # Check for stage execution time metric
        timing_calls = [call for call in metrics.record_timing.call_args_list]
        assert any("stage_execution_time_ms" in str(call) for call in timing_calls)

        # Check for workflow execution time metric
        assert any("workflow_execution_time_ms" in str(call) for call in timing_calls)

    @pytest.mark.asyncio
    async def test_signal_coordination(self, executor, signals):
        """Test that signals are sent during workflow execution."""

        async def test_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {"result": "success"}

        stage = WorkflowStage(
            stage_id="test_stage",
            stage_number=1,
            stage_name="Test Stage",
            steps=[
                WorkflowStep(
                    step_id="test_step",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=test_executor,
                )
            ],
        )

        # Execute workflow
        await executor.execute_workflow(
            workflow_id="test-workflow-9",
            stages=[stage],
        )

        # Verify signals were sent
        signals.signal_coordination_event.assert_called()

        # Check for stage completion signal
        signal_calls = signals.signal_coordination_event.call_args_list
        assert any(
            call.kwargs.get("event_type") == "stage_completed" for call in signal_calls
        )

    @pytest.mark.asyncio
    async def test_workflow_state_management(self, executor, state):
        """Test that workflow state is managed in ThreadSafeState."""

        async def test_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {"result": "success"}

        stage = WorkflowStage(
            stage_id="test_stage",
            stage_number=1,
            stage_name="Test Stage",
            steps=[
                WorkflowStep(
                    step_id="test_step",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=test_executor,
                )
            ],
        )

        workflow_id = "test-workflow-10"

        # Execute workflow
        result = await executor.execute_workflow(
            workflow_id=workflow_id,
            stages=[stage],
        )

        # Debug: Check all keys in state
        all_keys = state.keys()
        print(f"\nAll state keys: {all_keys}")
        print(f"Workflow result status: {result.status}")
        print(f"Executor state keys: {executor.state.keys()}")

        # Verify workflow state (use executor's state, not fixture state)
        status = executor.state.get(f"workflow_{workflow_id}_status")
        assert status == "completed"

        start_time = executor.state.get(f"workflow_{workflow_id}_start_time")
        assert start_time is not None

        end_time = executor.state.get(f"workflow_{workflow_id}_end_time")
        assert end_time is not None

    @pytest.mark.asyncio
    async def test_stage_sequence_validation(self, executor):
        """Test validation of stage sequence."""

        async def test_executor(context: dict[str, Any]) -> dict[str, Any]:
            return {}

        # Create stages with invalid sequence (2, 1 instead of 1, 2)
        stage1 = WorkflowStage(
            stage_id="stage2",
            stage_number=2,
            stage_name="Stage 2",
            steps=[
                WorkflowStep(
                    step_id="step2",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=test_executor,
                )
            ],
        )

        stage2 = WorkflowStage(
            stage_id="stage1",
            stage_number=1,
            stage_name="Stage 1",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    step_type=EnumStepType.GENERATE_MODEL,
                    dependencies=[],
                    input_data={},
                    executor=test_executor,
                )
            ],
        )

        # Execute workflow (should auto-sort and execute correctly)
        result = await executor.execute_workflow(
            workflow_id="test-workflow-11",
            stages=[stage1, stage2],  # Out of order
        )

        # Should succeed because executor sorts stages
        assert result.status == EnumStageStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_empty_stages_validation(self, executor):
        """Test validation with empty stages list."""
        with pytest.raises(ValueError, match="stages list cannot be empty"):
            await executor.execute_workflow(
                workflow_id="test-workflow-12",
                stages=[],
            )

    @pytest.mark.asyncio
    async def test_six_phase_code_generation_pipeline(self, executor):
        """Test complete 6-phase code generation pipeline."""
        phase_execution_order = []

        async def phase_executor(context: dict[str, Any]) -> dict[str, Any]:
            # Get step_type from task_metadata (set when adding task to scheduler)
            task_metadata = context.get("task_metadata", {})
            phase = task_metadata.get("step_type", "unknown")
            phase_execution_order.append(phase)
            await asyncio.sleep(0.05)  # Simulate work
            return {"phase": phase, "result": "success"}

        # Define 6 phases
        phases = [
            (1, "parse", EnumStepType.PARSE_CONTRACT, []),
            (2, "models", EnumStepType.GENERATE_MODEL, ["parse"]),
            (3, "validators", EnumStepType.GENERATE_VALIDATOR, ["models"]),
            (4, "tests", EnumStepType.GENERATE_TEST, ["validators"]),
            (5, "quality", EnumStepType.VALIDATE_QUALITY, ["tests"]),
            (6, "package", EnumStepType.PACKAGE_NODE, ["quality"]),
        ]

        stages = []
        for stage_number, stage_id, step_type, dependencies in phases:
            # Create 3 parallel steps per stage (simulating 3 contracts)
            steps = [
                WorkflowStep(
                    step_id=f"{stage_id}_{i}",
                    step_type=step_type,
                    dependencies=[],
                    input_data={},
                    executor=phase_executor,
                )
                for i in range(3)
            ]

            stage = WorkflowStage(
                stage_id=stage_id,
                stage_number=stage_number,
                stage_name=stage_id.capitalize(),
                steps=steps,
                dependencies=dependencies,
                parallel=True,
            )
            stages.append(stage)

        # Execute complete pipeline
        result = await executor.execute_workflow(
            workflow_id="codegen-pipeline",
            stages=stages,
        )

        # Verify results
        assert result.status == EnumStageStatus.COMPLETED
        assert result.total_stages == 6
        assert result.successful_stages == 6
        assert result.total_steps == 18  # 6 stages * 3 steps each
        assert result.successful_steps == 18

        # Verify execution order (all parse before models, etc.)
        parse_indices = [
            i
            for i, phase in enumerate(phase_execution_order)
            if phase == "parse_contract"
        ]
        model_indices = [
            i
            for i, phase in enumerate(phase_execution_order)
            if phase == "generate_model"
        ]

        assert max(parse_indices) < min(model_indices)

        # Verify speedup
        # With 6 stages * 3 steps * 50ms = 900ms sequential
        # Parallel should be ~300ms (stages sequential, steps parallel)
        # Expected speedup: ~2-3x
        assert result.overall_speedup >= 2.0
        print(f"\n6-Phase Pipeline Results:")
        print(f"  Total duration: {result.total_duration_ms:.2f}ms")
        print(f"  Overall speedup: {result.overall_speedup:.2f}x")
        print(f"  Success rate: {result.success_rate:.1%}")

        # Verify performance target (<5s for typical contract)
        assert result.total_duration_ms < 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

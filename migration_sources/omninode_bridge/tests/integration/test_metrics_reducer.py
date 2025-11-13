#!/usr/bin/env python3
"""
Integration tests for Code Generation Metrics Reducer.

Tests the complete metrics aggregation pipeline:
- Event processing
- Aggregation strategies
- FSM state tracking
- PostgreSQL persistence
- Performance requirements

ONEX v2.0 Test Compliance:
- Contract-driven test scenarios
- Performance validation
- Quality gates verification
"""

from uuid import uuid4

import pytest

from src.omninode_bridge.reducers.codegen_metrics_aggregator import (
    CodegenMetricsAggregator,
    EnumAggregationType,
    EnumDomain,
    EnumNodeType,
    ModelNodeGenerationCompleted,
    ModelNodeGenerationFailed,
    ModelNodeGenerationStageCompleted,
    ModelNodeGenerationStarted,
)
from src.omninode_bridge.reducers.fsm_state_tracker import (
    EnumTransitionEvent,
    EnumWorkflowState,
    FSMStateTracker,
)


class TestCodegenMetricsAggregator:
    """Test suite for CodegenMetricsAggregator."""

    @pytest.fixture
    async def aggregator(self):
        """Create aggregator instance for testing."""
        return CodegenMetricsAggregator(
            max_buffer_size=100,
            flush_interval_seconds=60,
            enable_postgres_persistence=False,
        )

    @pytest.mark.asyncio
    async def test_process_generation_workflow(self, aggregator):
        """Test complete generation workflow processing."""
        workflow_id = uuid4()

        # Start event
        start_event = ModelNodeGenerationStarted(
            workflow_id=workflow_id,
            node_type=EnumNodeType.EFFECT,
            domain=EnumDomain.API,
        )
        await aggregator.process_event(start_event)

        # Stage completed events
        for stage_num in range(1, 7):
            stage_event = ModelNodeGenerationStageCompleted(
                workflow_id=workflow_id,
                stage_name=f"stage_{stage_num}",
                stage_number=stage_num,
                duration_ms=100,
                success=True,
                tokens_consumed=100,
                cost_usd=0.01,
            )
            await aggregator.process_event(stage_event)

        # Completion event
        completion_event = ModelNodeGenerationCompleted(
            workflow_id=workflow_id,
            node_type=EnumNodeType.EFFECT,
            domain=EnumDomain.API,
            total_duration_seconds=1.5,
            quality_score=0.85,
            total_tokens=600,
            total_cost_usd=0.06,
        )
        await aggregator.process_event(completion_event)

        # Verify metrics
        metrics = await aggregator.get_metrics()
        assert (
            metrics["events_processed_total"] == 8
        )  # 1 start + 6 stages + 1 completion
        assert metrics["active_workflows"] == 0  # Workflow completed

    @pytest.mark.asyncio
    async def test_node_type_aggregation(self, aggregator):
        """Test NODE_TYPE_GROUPING aggregation strategy."""
        # Generate events for different node types
        for node_type in [
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
        ]:
            for i in range(3):
                event = ModelNodeGenerationCompleted(
                    workflow_id=uuid4(),
                    node_type=node_type,
                    total_duration_seconds=2.0,
                    quality_score=0.8,
                    total_tokens=1000,
                    total_cost_usd=0.10,
                )
                await aggregator.process_event(event)

        # Force flush
        await aggregator._flush_buffers()

        # Verify buffer cleared
        assert len(aggregator._buffers[EnumAggregationType.NODE_TYPE_GROUPING]) == 0

    @pytest.mark.asyncio
    async def test_quality_bucket_aggregation(self, aggregator):
        """Test QUALITY_BUCKETS aggregation strategy."""
        # Generate events with different quality scores
        quality_scores = [0.4, 0.7, 0.9]  # Low, Medium, High

        for quality_score in quality_scores:
            event = ModelNodeGenerationCompleted(
                workflow_id=uuid4(),
                node_type=EnumNodeType.EFFECT,
                total_duration_seconds=1.0,
                quality_score=quality_score,
                total_tokens=500,
                total_cost_usd=0.05,
            )
            await aggregator.process_event(event)

        # Force flush
        await aggregator._flush_buffers()

        # Verify aggregations completed
        assert aggregator._aggregations_completed > 0

    @pytest.mark.asyncio
    async def test_time_window_aggregation(self, aggregator):
        """Test TIME_WINDOW aggregation strategy."""
        # Generate events
        for i in range(5):
            event = ModelNodeGenerationCompleted(
                workflow_id=uuid4(),
                node_type=EnumNodeType.COMPUTE,
                total_duration_seconds=1.5,
                quality_score=0.85,
                total_tokens=800,
                total_cost_usd=0.08,
            )
            await aggregator.process_event(event)

        # Force flush
        await aggregator._flush_buffers()

        # Verify all window types processed
        metrics = await aggregator.get_metrics()
        assert metrics["aggregations_completed"] > 0

    @pytest.mark.asyncio
    async def test_domain_aggregation(self, aggregator):
        """Test DOMAIN_GROUPING aggregation strategy."""
        # Generate events for different domains
        for domain in [EnumDomain.API, EnumDomain.ML, EnumDomain.DATA]:
            for i in range(2):
                event = ModelNodeGenerationCompleted(
                    workflow_id=uuid4(),
                    node_type=EnumNodeType.EFFECT,
                    domain=domain,
                    total_duration_seconds=2.0,
                    quality_score=0.8,
                    total_tokens=1000,
                    total_cost_usd=0.10,
                )
                await aggregator.process_event(event)

        # Force flush
        await aggregator._flush_buffers()

        # Verify aggregations
        assert aggregator._aggregations_completed > 0

    @pytest.mark.asyncio
    async def test_failure_handling(self, aggregator):
        """Test generation failure event handling."""
        workflow_id = uuid4()

        # Start event
        start_event = ModelNodeGenerationStarted(
            workflow_id=workflow_id,
            node_type=EnumNodeType.EFFECT,
        )
        await aggregator.process_event(start_event)

        # Failure event
        failure_event = ModelNodeGenerationFailed(
            workflow_id=workflow_id,
            node_type=EnumNodeType.EFFECT,
            failed_stage="validation",
            error_message="Quality score below threshold",
        )
        await aggregator.process_event(failure_event)

        # Verify workflow removed from active tracking
        metrics = await aggregator.get_metrics()
        assert metrics["active_workflows"] == 0

    @pytest.mark.asyncio
    async def test_performance_throughput(self, aggregator):
        """Test aggregation throughput (>1000 events/second target)."""
        import time

        num_events = 1000
        start_time = time.perf_counter()

        # Generate 1000 events
        for i in range(num_events):
            event = ModelNodeGenerationCompleted(
                workflow_id=uuid4(),
                node_type=EnumNodeType.COMPUTE,
                total_duration_seconds=1.0,
                quality_score=0.8,
                total_tokens=500,
                total_cost_usd=0.05,
            )
            await aggregator.process_event(event)

        duration = time.perf_counter() - start_time
        throughput = num_events / duration

        # Verify throughput meets target
        assert (
            throughput > 1000
        ), f"Throughput {throughput:.0f} events/sec < 1000 target"

    @pytest.mark.asyncio
    async def test_aggregation_latency(self, aggregator):
        """Test aggregation latency (<100ms for 1000 items target)."""
        import time

        # Fill buffer with 1000 events
        for i in range(1000):
            event = ModelNodeGenerationCompleted(
                workflow_id=uuid4(),
                node_type=EnumNodeType.EFFECT,
                total_duration_seconds=1.5,
                quality_score=0.85,
                total_tokens=600,
                total_cost_usd=0.06,
            )
            await aggregator.process_event(event)

        # Measure flush time
        start_time = time.perf_counter()
        await aggregator._flush_buffers()
        flush_duration_ms = (time.perf_counter() - start_time) * 1000

        # Verify latency meets target
        assert (
            flush_duration_ms < 100
        ), f"Flush latency {flush_duration_ms:.2f}ms > 100ms target"


class TestFSMStateTracker:
    """Test suite for FSMStateTracker."""

    @pytest.fixture
    async def tracker(self):
        """Create FSM tracker instance for testing."""
        return FSMStateTracker(enable_postgres_persistence=False)

    @pytest.mark.asyncio
    async def test_workflow_initialization(self, tracker):
        """Test workflow initialization."""
        workflow_id = uuid4()

        success = await tracker.initialize_workflow(
            workflow_id=workflow_id,
            initial_state=EnumWorkflowState.PENDING,
            metadata={"node_type": "effect"},
        )

        assert success
        workflow = await tracker.get_workflow_state(workflow_id)
        assert workflow is not None
        assert workflow.current_state == EnumWorkflowState.PENDING

    @pytest.mark.asyncio
    async def test_valid_state_transitions(self, tracker):
        """Test valid FSM state transitions."""
        workflow_id = uuid4()
        await tracker.initialize_workflow(workflow_id)

        # PENDING -> ANALYZING
        success, error = await tracker.transition_state(
            workflow_id=workflow_id,
            to_state=EnumWorkflowState.ANALYZING,
            event=EnumTransitionEvent.START_ANALYSIS,
            metadata={"has_requirements": True},
        )
        assert success

        # ANALYZING -> GENERATING
        success, error = await tracker.transition_state(
            workflow_id=workflow_id,
            to_state=EnumWorkflowState.GENERATING,
            event=EnumTransitionEvent.START_GENERATION,
            metadata={"has_analysis_results": True},
        )
        assert success

        # GENERATING -> VALIDATING
        success, error = await tracker.transition_state(
            workflow_id=workflow_id,
            to_state=EnumWorkflowState.VALIDATING,
            event=EnumTransitionEvent.START_VALIDATION,
            metadata={"has_generated_code": True},
        )
        assert success

        # VALIDATING -> COMPLETED
        success, error = await tracker.transition_state(
            workflow_id=workflow_id,
            to_state=EnumWorkflowState.COMPLETED,
            event=EnumTransitionEvent.COMPLETE_WORKFLOW,
            metadata={"validation_passed": True},
        )
        assert success

        # Verify final state
        workflow = await tracker.get_workflow_state(workflow_id)
        assert workflow.current_state == EnumWorkflowState.COMPLETED
        assert workflow.transition_count == 4

    @pytest.mark.asyncio
    async def test_invalid_state_transition(self, tracker):
        """Test invalid state transition rejection."""
        workflow_id = uuid4()
        await tracker.initialize_workflow(workflow_id)

        # Try invalid transition: PENDING -> COMPLETED
        success, error = await tracker.transition_state(
            workflow_id=workflow_id,
            to_state=EnumWorkflowState.COMPLETED,
            event=EnumTransitionEvent.COMPLETE_WORKFLOW,
        )

        assert not success
        assert error is not None
        assert "Invalid transition" in error

    @pytest.mark.asyncio
    async def test_guard_conditions(self, tracker):
        """Test guard condition validation."""
        workflow_id = uuid4()
        await tracker.initialize_workflow(workflow_id)

        # Transition without required guard condition
        success, error = await tracker.transition_state(
            workflow_id=workflow_id,
            to_state=EnumWorkflowState.ANALYZING,
            event=EnumTransitionEvent.START_ANALYSIS,
            metadata={},  # Missing has_requirements
        )

        assert not success
        assert "Guard condition failed" in error

    @pytest.mark.asyncio
    async def test_transition_history(self, tracker):
        """Test transition history tracking."""
        workflow_id = uuid4()
        await tracker.initialize_workflow(workflow_id)

        # Perform transitions
        await tracker.transition_state(
            workflow_id,
            EnumWorkflowState.ANALYZING,
            EnumTransitionEvent.START_ANALYSIS,
            {"has_requirements": True},
        )
        await tracker.transition_state(
            workflow_id,
            EnumWorkflowState.GENERATING,
            EnumTransitionEvent.START_GENERATION,
            {"has_analysis_results": True},
        )

        # Get history
        history = await tracker.get_transition_history(workflow_id)
        assert len(history) == 2
        assert history[0].from_state == EnumWorkflowState.PENDING
        assert history[0].to_state == EnumWorkflowState.ANALYZING
        assert history[1].from_state == EnumWorkflowState.ANALYZING
        assert history[1].to_state == EnumWorkflowState.GENERATING

    @pytest.mark.asyncio
    async def test_failure_recovery(self, tracker):
        """Test failure and retry workflow."""
        workflow_id = uuid4()
        await tracker.initialize_workflow(workflow_id)

        # Transition to ANALYZING
        await tracker.transition_state(
            workflow_id,
            EnumWorkflowState.ANALYZING,
            EnumTransitionEvent.START_ANALYSIS,
            {"has_requirements": True},
        )

        # Fail
        success, _ = await tracker.transition_state(
            workflow_id,
            EnumWorkflowState.FAILED,
            EnumTransitionEvent.FAIL_WORKFLOW,
        )
        assert success

        # Retry
        success, _ = await tracker.transition_state(
            workflow_id,
            EnumWorkflowState.PENDING,
            EnumTransitionEvent.RETRY_WORKFLOW,
        )
        assert success

    @pytest.mark.asyncio
    async def test_performance_state_transition(self, tracker):
        """Test state transition performance (<10ms target)."""
        import time

        workflow_id = uuid4()
        await tracker.initialize_workflow(workflow_id)

        # Measure transition time
        start_time = time.perf_counter()
        await tracker.transition_state(
            workflow_id,
            EnumWorkflowState.ANALYZING,
            EnumTransitionEvent.START_ANALYSIS,
            {"has_requirements": True},
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Verify performance target
        assert duration_ms < 10, f"Transition time {duration_ms:.2f}ms > 10ms target"

    @pytest.mark.asyncio
    async def test_cleanup_terminal_workflows(self, tracker):
        """Test cleanup of terminal workflows."""
        # Create and complete workflows
        for i in range(5):
            workflow_id = uuid4()
            await tracker.initialize_workflow(workflow_id)
            await tracker.transition_state(
                workflow_id,
                EnumWorkflowState.ANALYZING,
                EnumTransitionEvent.START_ANALYSIS,
                {"has_requirements": True},
            )
            await tracker.transition_state(
                workflow_id,
                EnumWorkflowState.FAILED,
                EnumTransitionEvent.FAIL_WORKFLOW,
            )

        # Cleanup (older than 0 hours = all)
        removed = await tracker.cleanup_terminal_workflows(older_than_hours=0)
        assert removed == 5


@pytest.mark.asyncio
async def test_end_to_end_integration():
    """Test end-to-end integration of metrics aggregation and FSM tracking."""
    aggregator = CodegenMetricsAggregator()
    tracker = FSMStateTracker()

    workflow_id = uuid4()

    # Initialize workflow in FSM
    await tracker.initialize_workflow(workflow_id)

    # Start generation
    start_event = ModelNodeGenerationStarted(
        workflow_id=workflow_id,
        node_type=EnumNodeType.EFFECT,
        domain=EnumDomain.API,
    )
    await aggregator.process_event(start_event)
    await tracker.transition_state(
        workflow_id,
        EnumWorkflowState.ANALYZING,
        EnumTransitionEvent.START_ANALYSIS,
        {"has_requirements": True},
    )

    # Complete stages
    await tracker.transition_state(
        workflow_id,
        EnumWorkflowState.GENERATING,
        EnumTransitionEvent.START_GENERATION,
        {"has_analysis_results": True},
    )
    await tracker.transition_state(
        workflow_id,
        EnumWorkflowState.VALIDATING,
        EnumTransitionEvent.START_VALIDATION,
        {"has_generated_code": True},
    )

    # Complete generation
    completion_event = ModelNodeGenerationCompleted(
        workflow_id=workflow_id,
        node_type=EnumNodeType.EFFECT,
        domain=EnumDomain.API,
        total_duration_seconds=3.5,
        quality_score=0.88,
        total_tokens=1200,
        total_cost_usd=0.12,
    )
    await aggregator.process_event(completion_event)
    await tracker.transition_state(
        workflow_id,
        EnumWorkflowState.COMPLETED,
        EnumTransitionEvent.COMPLETE_WORKFLOW,
        {"validation_passed": True},
    )

    # Verify final state
    workflow = await tracker.get_workflow_state(workflow_id)
    assert workflow.current_state == EnumWorkflowState.COMPLETED

    aggregator_metrics = await aggregator.get_metrics()
    assert aggregator_metrics["events_processed_total"] == 2

    tracker_metrics = await tracker.get_metrics()
    assert tracker_metrics["total_transitions"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

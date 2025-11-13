"""
Tests for MetricsAggregator - Pure Function Testing.

These tests validate the pure aggregation logic without any I/O dependencies.
All tests use pure functions and fixture data for deterministic results.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from omninode_bridge.events.models.codegen_events import (
    ModelEventCodegenCompleted,
    ModelEventCodegenFailed,
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
)
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.aggregator import (
    MetricsAggregator,
)
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.models.enum_metrics_window import (
    EnumMetricsWindow,
)


class TestMetricsAggregator:
    """Test suite for MetricsAggregator pure functions."""

    @pytest.fixture
    def base_timestamp(self) -> datetime:
        """Base timestamp for test events."""
        return datetime.now(UTC)

    @pytest.fixture
    def workflow_id(self) -> UUID:
        """Workflow ID for test events."""
        return uuid4()

    @pytest.fixture
    def started_event(
        self, base_timestamp: datetime, workflow_id: UUID
    ) -> ModelEventCodegenStarted:
        """Create a NODE_GENERATION_STARTED event."""
        return ModelEventCodegenStarted(
            correlation_id=uuid4(),
            event_id=uuid4(),
            timestamp=base_timestamp,
            workflow_id=workflow_id,
            orchestrator_node_id=uuid4(),
            prompt="Test node generation",
            output_directory="/test",
        )

    @pytest.fixture
    def stage_event(
        self, base_timestamp: datetime, workflow_id: UUID
    ) -> ModelEventCodegenStageCompleted:
        """Create a NODE_GENERATION_STAGE_COMPLETED event."""
        return ModelEventCodegenStageCompleted(
            correlation_id=uuid4(),
            event_id=uuid4(),
            timestamp=base_timestamp + timedelta(seconds=5),
            workflow_id=workflow_id,
            stage_name="prompt_parsing",
            stage_number=1,
            duration_seconds=2.5,
            success=True,
        )

    @pytest.fixture
    def completed_event(
        self, base_timestamp: datetime, workflow_id: UUID
    ) -> ModelEventCodegenCompleted:
        """Create a NODE_GENERATION_COMPLETED event."""
        return ModelEventCodegenCompleted(
            correlation_id=uuid4(),
            event_id=uuid4(),
            timestamp=base_timestamp + timedelta(seconds=60),
            workflow_id=workflow_id,
            total_duration_seconds=53.5,
            generated_files=["/test/node.py"],
            node_type="effect",
            service_name="test_service",
            quality_score=0.92,
            test_coverage=0.85,
            complexity_score=5.2,
            patterns_applied=["pattern1", "pattern2"],
            intelligence_sources=["qdrant"],
            primary_model="gemini-2.5-flash",
            total_tokens=1500,
            total_cost_usd=0.03,
            contract_yaml="...",
            node_module="/test/node.py",
            models=[],
            enums=[],
            tests=[],
        )

    @pytest.fixture
    def failed_event(
        self, base_timestamp: datetime, workflow_id: UUID
    ) -> ModelEventCodegenFailed:
        """Create a NODE_GENERATION_FAILED event."""
        return ModelEventCodegenFailed(
            correlation_id=uuid4(),
            event_id=uuid4(),
            timestamp=base_timestamp + timedelta(seconds=30),
            workflow_id=workflow_id,
            failed_stage="code_generation",
            partial_duration_seconds=25.0,
            error_code="CODEGEN_ERROR",
            error_message="Failed to generate code",
            error_context={},
        )

    def test_aggregate_empty_events(self):
        """Test aggregation with no events."""
        aggregator = MetricsAggregator()
        result = aggregator.aggregate_events(
            events=[], window_type=EnumMetricsWindow.HOURLY
        )

        assert result.total_generations == 0
        assert result.successful_generations == 0
        assert result.failed_generations == 0
        assert result.avg_duration_seconds == 0.0
        assert result.events_processed == 0

    def test_aggregate_single_completed_event(self, started_event, completed_event):
        """Test aggregation with single completed workflow."""
        aggregator = MetricsAggregator()
        events = [started_event, completed_event]

        result = aggregator.aggregate_events(
            events=events, window_type=EnumMetricsWindow.HOURLY
        )

        # Performance metrics
        assert result.total_generations == 1
        assert result.successful_generations == 1
        assert result.failed_generations == 0
        assert result.avg_duration_seconds == 53.5
        assert result.p50_duration_seconds == 53.5
        assert result.min_duration_seconds == 53.5
        assert result.max_duration_seconds == 53.5

        # Quality metrics
        assert result.avg_quality_score == 0.92
        assert result.avg_test_coverage == 0.85
        assert result.avg_complexity_score == 5.2

        # Cost metrics
        assert result.total_tokens == 1500
        assert result.total_cost_usd == 0.03
        assert result.avg_cost_per_generation == 0.03

        # Intelligence usage
        assert result.intelligence_enabled_count == 1
        assert result.avg_patterns_applied == 2.0

        # Workflow tracking
        assert result.workflow_ids_tracked == 1
        assert result.events_processed == 2

    def test_aggregate_multiple_workflows(self, base_timestamp):
        """Test aggregation with multiple workflows."""
        aggregator = MetricsAggregator()

        # Create 10 completed workflows with varying metrics
        events = []
        for i in range(10):
            workflow_id = uuid4()

            started = ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=i),
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt=f"Test node {i}",
                output_directory="/test",
            )

            completed = ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=60 + i),
                workflow_id=workflow_id,
                total_duration_seconds=50.0 + i,  # Varying durations
                generated_files=[f"/test/node_{i}.py"],
                node_type="effect",
                service_name=f"test_service_{i}",
                quality_score=0.90 + (i * 0.01),  # 0.90-0.99
                test_coverage=0.80 + (i * 0.01),
                primary_model="gemini-2.5-flash",
                total_tokens=1000 + (i * 100),  # 1000-1900
                total_cost_usd=0.02 + (i * 0.001),  # 0.02-0.029
                patterns_applied=[f"pattern{i}"],
                contract_yaml="...",
                node_module=f"/test/node_{i}.py",
                models=[],
                enums=[],
                tests=[],
            )

            events.extend([started, completed])

        result = aggregator.aggregate_events(
            events=events, window_type=EnumMetricsWindow.DAILY
        )

        # Performance metrics
        assert result.total_generations == 10
        assert result.successful_generations == 10
        assert result.failed_generations == 0

        # Verify duration statistics
        assert 50.0 <= result.avg_duration_seconds <= 59.0
        assert result.min_duration_seconds == 50.0
        assert result.max_duration_seconds == 59.0

        # Verify quality metrics
        assert 0.90 <= result.avg_quality_score <= 0.99

        # Verify cost metrics
        assert result.total_tokens == sum(1000 + (i * 100) for i in range(10))
        assert 0.02 <= result.avg_cost_per_generation <= 0.03

        # Workflow tracking
        assert result.workflow_ids_tracked == 10
        assert result.events_processed == 20

    def test_aggregate_with_failures(self, base_timestamp):
        """Test aggregation with both successful and failed workflows."""
        aggregator = MetricsAggregator()

        events = []

        # 7 successful workflows
        for i in range(7):
            workflow_id = uuid4()

            started = ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=i),
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt=f"Test node {i}",
                output_directory="/test",
            )

            completed = ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=60 + i),
                workflow_id=workflow_id,
                total_duration_seconds=55.0,
                generated_files=[f"/test/node_{i}.py"],
                node_type="effect",
                service_name=f"test_service_{i}",
                quality_score=0.92,
                primary_model="gemini-2.5-flash",
                total_tokens=1500,
                total_cost_usd=0.03,
                contract_yaml="...",
                node_module=f"/test/node_{i}.py",
                models=[],
                enums=[],
                tests=[],
            )

            events.extend([started, completed])

        # 3 failed workflows
        for i in range(3):
            workflow_id = uuid4()

            started = ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=10 + i),
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt=f"Test failed node {i}",
                output_directory="/test",
            )

            failed = ModelEventCodegenFailed(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=40 + i),
                workflow_id=workflow_id,
                failed_stage="code_generation",
                partial_duration_seconds=30.0,
                error_code="CODEGEN_ERROR",
                error_message="Test failure",
                error_context={},
            )

            events.extend([started, failed])

        result = aggregator.aggregate_events(
            events=events, window_type=EnumMetricsWindow.HOURLY
        )

        # Verify success/failure counts
        assert result.total_generations == 10
        assert result.successful_generations == 7
        assert result.failed_generations == 3

        # Verify metrics computed only from successful workflows
        assert result.avg_duration_seconds == 55.0
        assert result.avg_quality_score == 0.92

    def test_aggregate_stage_performance(self, base_timestamp, workflow_id):
        """Test aggregation of stage-level performance."""
        aggregator = MetricsAggregator()

        # Create events with multiple stages
        events = [
            ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp,
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt="Test node",
                output_directory="/test",
            ),
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=2),
                workflow_id=workflow_id,
                stage_name="prompt_parsing",
                stage_number=1,
                duration_seconds=2.0,
                success=True,
            ),
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=12),
                workflow_id=workflow_id,
                stage_name="code_generation",
                stage_number=4,
                duration_seconds=10.0,
                success=True,
            ),
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=15),
                workflow_id=workflow_id,
                stage_name="validation",
                stage_number=6,
                duration_seconds=3.0,
                success=True,
            ),
        ]

        result = aggregator.aggregate_events(
            events=events, window_type=EnumMetricsWindow.HOURLY
        )

        # Verify stage durations computed
        assert "prompt_parsing" in result.avg_stage_durations
        assert result.avg_stage_durations["prompt_parsing"] == 2.0
        assert result.avg_stage_durations["code_generation"] == 10.0
        assert result.avg_stage_durations["validation"] == 3.0

    def test_aggregate_model_metrics(self, base_timestamp):
        """Test aggregation of per-model metrics."""
        aggregator = MetricsAggregator()

        # Create workflows using different models
        events = []

        # 5 workflows with gemini
        for i in range(5):
            workflow_id = uuid4()

            started = ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=i),
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt=f"Test gemini {i}",
                output_directory="/test",
            )

            completed = ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=60 + i),
                workflow_id=workflow_id,
                total_duration_seconds=50.0,
                generated_files=[f"/test/node_{i}.py"],
                node_type="effect",
                service_name=f"test_service_{i}",
                quality_score=0.92,
                primary_model="gemini-2.5-flash",
                total_tokens=1500,
                total_cost_usd=0.03,
                contract_yaml="...",
                node_module=f"/test/node_{i}.py",
                models=[],
                enums=[],
                tests=[],
            )

            events.extend([started, completed])

        # 3 workflows with claude
        for i in range(3):
            workflow_id = uuid4()

            started = ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=10 + i),
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt=f"Test claude {i}",
                output_directory="/test",
            )

            completed = ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=80 + i),
                workflow_id=workflow_id,
                total_duration_seconds=65.0,
                generated_files=[f"/test/node_claude_{i}.py"],
                node_type="orchestrator",
                service_name=f"test_service_claude_{i}",
                quality_score=0.95,
                primary_model="claude-3.5-sonnet",
                total_tokens=2500,
                total_cost_usd=0.08,
                contract_yaml="...",
                node_module=f"/test/node_claude_{i}.py",
                models=[],
                enums=[],
                tests=[],
            )

            events.extend([started, completed])

        result = aggregator.aggregate_events(
            events=events, window_type=EnumMetricsWindow.DAILY
        )

        # Verify per-model metrics
        assert "gemini-2.5-flash" in result.model_metrics
        assert "claude-3.5-sonnet" in result.model_metrics

        gemini_metrics = result.model_metrics["gemini-2.5-flash"]
        assert gemini_metrics["total_generations"] == 5
        assert gemini_metrics["avg_duration_seconds"] == 50.0
        assert gemini_metrics["avg_quality_score"] == 0.92

        claude_metrics = result.model_metrics["claude-3.5-sonnet"]
        assert claude_metrics["total_generations"] == 3
        assert claude_metrics["avg_duration_seconds"] == 65.0
        assert claude_metrics["avg_quality_score"] == 0.95

    def test_aggregate_node_type_metrics(self, base_timestamp):
        """Test aggregation of per-node-type metrics."""
        aggregator = MetricsAggregator()

        # Create workflows for different node types
        events = []

        # 4 effect nodes
        for i in range(4):
            workflow_id = uuid4()

            started = ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=i),
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt=f"Test effect {i}",
                output_directory="/test",
            )

            completed = ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=50 + i),
                workflow_id=workflow_id,
                total_duration_seconds=45.0,
                generated_files=[f"/test/node_{i}.py"],
                node_type="effect",
                service_name=f"test_service_{i}",
                quality_score=0.90,
                primary_model="gemini-2.5-flash",
                total_tokens=1200,
                total_cost_usd=0.025,
                contract_yaml="...",
                node_module=f"/test/node_{i}.py",
                models=[],
                enums=[],
                tests=[],
            )

            events.extend([started, completed])

        # 3 orchestrator nodes
        for i in range(3):
            workflow_id = uuid4()

            started = ModelEventCodegenStarted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=10 + i),
                workflow_id=workflow_id,
                orchestrator_node_id=uuid4(),
                prompt=f"Test orchestrator {i}",
                output_directory="/test",
            )

            completed = ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                timestamp=base_timestamp + timedelta(seconds=80 + i),
                workflow_id=workflow_id,
                total_duration_seconds=70.0,
                generated_files=[f"/test/node_orch_{i}.py"],
                node_type="orchestrator",
                service_name=f"test_service_orch_{i}",
                quality_score=0.93,
                primary_model="gemini-2.5-flash",
                total_tokens=2000,
                total_cost_usd=0.04,
                contract_yaml="...",
                node_module=f"/test/node_orch_{i}.py",
                models=[],
                enums=[],
                tests=[],
            )

            events.extend([started, completed])

        result = aggregator.aggregate_events(
            events=events, window_type=EnumMetricsWindow.HOURLY
        )

        # Verify per-node-type metrics
        assert "effect" in result.node_type_metrics
        assert "orchestrator" in result.node_type_metrics

        effect_metrics = result.node_type_metrics["effect"]
        assert effect_metrics["total_generations"] == 4
        assert effect_metrics["avg_duration_seconds"] == 45.0
        assert effect_metrics["avg_quality_score"] == 0.90

        orch_metrics = result.node_type_metrics["orchestrator"]
        assert orch_metrics["total_generations"] == 3
        assert orch_metrics["avg_duration_seconds"] == 70.0
        assert orch_metrics["avg_quality_score"] == 0.93

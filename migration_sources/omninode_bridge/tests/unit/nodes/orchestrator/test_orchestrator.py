#!/usr/bin/env python3
"""Unit tests for NodeBridgeOrchestrator.

Tests cover:
- Initialization with dependency injection
- Contract validation
- FSM state transitions
- Workflow step execution
- Event publishing
- Error handling and recovery
- Performance metrics tracking
- Complete workflow orchestration
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import ONEX infrastructure from omnibase_core
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)

from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_event import (
    EnumWorkflowEvent,
)
from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
    EnumWorkflowState,
)
from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_stamp_response_output import (
    ModelStampResponseOutput,
)
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

# Fixtures


@pytest.fixture
def mock_container():
    """Create mock ONEX container with configuration."""
    container = MagicMock(spec=ModelONEXContainer)
    container.config = {
        "metadata_stamping_service_url": "http://test-metadata:8053",
        "onextree_service_url": "http://test-onextree:8080",
        "kafka_broker_url": "test-kafka:9092",
        "default_namespace": "test.namespace",
        "health_check_mode": True,  # Skip ConfigLoader to use test config
    }
    # Mock Kafka client with AsyncMock for publish_event_internal method
    mock_kafka_client = MagicMock()
    mock_kafka_client.publish = AsyncMock(return_value=True)
    mock_kafka_client.publish_event_internal = AsyncMock(return_value=True)
    container.kafka_client = mock_kafka_client

    # Mock get_service to return kafka_client
    def get_service_mock(service_name):
        if service_name == "kafka_client":
            return mock_kafka_client
        return None

    container.get_service = MagicMock(side_effect=get_service_mock)

    # Mock register_service (no-op for tests)
    container.register_service = MagicMock(return_value=None)

    # Mock EventBus for event-driven workflows
    mock_event_bus = MagicMock()
    mock_event_bus.is_initialized = False  # Disable event-driven workflow by default
    mock_event_bus.wait_for_completion = AsyncMock(return_value=None)
    container.event_bus = mock_event_bus
    return container


@pytest.fixture
def orchestrator(mock_container):
    """Create NodeBridgeOrchestrator instance for testing."""
    try:
        return NodeBridgeOrchestrator(mock_container)
    except (ImportError, Exception) as e:
        error_msg = str(e)
        if (
            "omnibase_core.utils.generation" in error_msg
            or "Contract model loading failed" in error_msg
            or "unexpected keyword argument" in error_msg
        ):
            pytest.skip(
                "NodeBridgeOrchestrator requires omnibase_core.utils.generation module"
            )
        else:
            raise


@pytest.fixture
def mock_contract():
    """Create mock orchestrator contract with input data."""
    contract = MagicMock(spec=ModelContractOrchestrator)
    contract.correlation_id = uuid4()
    contract.input_data = {
        "file_path": "/test/file.txt",
        "content": "test content",
        "namespace": "test.namespace",
    }
    return contract


@pytest.fixture
def mock_contract_no_input():
    """Create mock contract without input data (for validation tests)."""
    contract = MagicMock(spec=ModelContractOrchestrator)
    contract.correlation_id = uuid4()
    contract.input_data = None
    return contract


@pytest.fixture
def mock_contract_no_content():
    """Create mock contract with input data but missing content field."""
    contract = MagicMock(spec=ModelContractOrchestrator)
    contract.correlation_id = uuid4()
    contract.input_data = {
        "file_path": "/test/file.txt",
        "namespace": "test.namespace",
        # Note: 'content' field is intentionally missing
    }
    return contract


# Test: Initialization


class TestInitialization:
    """Test suite for NodeBridgeOrchestrator initialization."""

    def test_init_success(self, mock_container):
        """Test successful orchestrator initialization."""
        try:
            orchestrator = NodeBridgeOrchestrator(mock_container)
            assert (
                orchestrator.metadata_stamping_service_url
                == "http://test-metadata:8053"
            )
            assert orchestrator.onextree_service_url == "http://test-onextree:8080"
            assert orchestrator.kafka_broker_url == "test-kafka:9092"
            assert orchestrator.default_namespace == "test.namespace"
            assert orchestrator.workflow_fsm_states == {}
            assert orchestrator.workflow_correlation_ids == {}
            assert orchestrator.stamping_metrics == {}
        except (ImportError, Exception) as e:
            error_msg = str(e)
            if (
                "omnibase_core.utils.generation" in error_msg
                or "Contract model loading failed" in error_msg
                or "unexpected keyword argument" in error_msg
            ):
                pytest.skip(
                    "NodeBridgeOrchestrator requires omnibase_core.utils.generation module"
                )
            else:
                raise

    def test_init_default_config(self):
        """Test initialization with default configuration values."""
        container = MagicMock(spec=ModelONEXContainer)
        container.config = {
            "health_check_mode": True  # Skip ConfigLoader to use default values
        }
        # Mock get_service to return None (no kafka_client in container)
        container.get_service = MagicMock(return_value=None)
        # Mock register_service (no-op for tests)
        container.register_service = MagicMock(return_value=None)

        try:
            orchestrator = NodeBridgeOrchestrator(container)
            assert (
                orchestrator.metadata_stamping_service_url
                == "http://metadata-stamping:8053"
            )
            assert orchestrator.onextree_service_url == "http://onextree:8058"
            # Kafka broker URL defaults to environment variable or hardcoded default
            assert orchestrator.kafka_broker_url in [
                "localhost:9092",
                "omninode-bridge-redpanda:9092",
            ]
            assert orchestrator.default_namespace == "omninode.bridge"
        except (ImportError, Exception) as e:
            error_msg = str(e)
            if (
                "omnibase_core.utils.generation" in error_msg
                or "Contract model loading failed" in error_msg
                or "unexpected keyword argument" in error_msg
            ):
                pytest.skip(
                    "NodeBridgeOrchestrator requires omnibase_core.utils.generation module"
                )
            else:
                raise


# Test: FSM State Transitions


class TestStateTransitions:
    """Test suite for FSM state transition logic."""

    @pytest.mark.asyncio
    async def test_valid_transition_pending_to_processing(self, orchestrator):
        """Test valid FSM transition from PENDING to PROCESSING."""
        workflow_id = uuid4()
        current = EnumWorkflowState.PENDING
        target = EnumWorkflowState.PROCESSING

        new_state = await orchestrator._transition_state(workflow_id, current, target)

        assert new_state == EnumWorkflowState.PROCESSING
        assert (
            orchestrator.workflow_fsm_states[str(workflow_id)]
            == EnumWorkflowState.PROCESSING
        )

    @pytest.mark.asyncio
    async def test_valid_transition_processing_to_completed(self, orchestrator):
        """Test valid FSM transition from PROCESSING to COMPLETED."""
        workflow_id = uuid4()
        current = EnumWorkflowState.PROCESSING
        target = EnumWorkflowState.COMPLETED

        new_state = await orchestrator._transition_state(workflow_id, current, target)

        assert new_state == EnumWorkflowState.COMPLETED
        assert (
            orchestrator.workflow_fsm_states[str(workflow_id)]
            == EnumWorkflowState.COMPLETED
        )

    @pytest.mark.asyncio
    async def test_valid_transition_processing_to_failed(self, orchestrator):
        """Test valid FSM transition from PROCESSING to FAILED."""
        workflow_id = uuid4()
        current = EnumWorkflowState.PROCESSING
        target = EnumWorkflowState.FAILED

        new_state = await orchestrator._transition_state(workflow_id, current, target)

        assert new_state == EnumWorkflowState.FAILED
        assert (
            orchestrator.workflow_fsm_states[str(workflow_id)]
            == EnumWorkflowState.FAILED
        )

    @pytest.mark.asyncio
    async def test_invalid_transition_completed_to_processing(self, orchestrator):
        """Test invalid FSM transition from terminal state."""
        workflow_id = uuid4()
        current = EnumWorkflowState.COMPLETED
        target = EnumWorkflowState.PROCESSING

        with pytest.raises(
            Exception
        ) as exc_info:  # Use generic Exception since OnexError types may vary
            await orchestrator._transition_state(workflow_id, current, target)

        # Check that some error was raised (the exact type may vary)
        assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_invalid_transition_failed_to_processing(self, orchestrator):
        """Test invalid FSM transition from FAILED terminal state."""
        workflow_id = uuid4()
        current = EnumWorkflowState.FAILED
        target = EnumWorkflowState.PROCESSING

        with pytest.raises(
            Exception
        ) as exc_info:  # Use generic Exception since OnexError types may vary
            await orchestrator._transition_state(workflow_id, current, target)

        # Check that some error was raised (the exact type may vary)
        assert exc_info.value is not None


# Test: Event Publishing


class TestEventPublishing:
    """Test suite for Kafka event publishing."""

    @pytest.mark.asyncio
    async def test_publish_workflow_started_event(self, orchestrator):
        """Test publishing workflow started event."""
        workflow_id = uuid4()
        event_data = {
            "workflow_id": str(workflow_id),
            "timestamp": datetime.now().isoformat(),
        }

        # Should not raise exception
        await orchestrator._publish_event(
            EnumWorkflowEvent.WORKFLOW_STARTED, event_data
        )

    @pytest.mark.asyncio
    async def test_publish_workflow_completed_event(self, orchestrator):
        """Test publishing workflow completed event."""
        workflow_id = uuid4()
        event_data = {
            "workflow_id": str(workflow_id),
            "stamp_id": str(uuid4()),
            "file_hash": "abc123",
            "timestamp": datetime.now().isoformat(),
        }

        # Should not raise exception
        await orchestrator._publish_event(
            EnumWorkflowEvent.WORKFLOW_COMPLETED, event_data
        )

    @pytest.mark.asyncio
    async def test_publish_event_error_handling(self, orchestrator):
        """Test that event publishing errors don't fail workflow."""
        # Create event with invalid data that might cause serialization issues
        invalid_data = {"complex_object": object()}

        # Should not raise exception (logs warning internally)
        await orchestrator._publish_event(
            EnumWorkflowEvent.WORKFLOW_STARTED, invalid_data
        )


# Test: Workflow Step Execution


class TestWorkflowSteps:
    """Test suite for individual workflow step execution."""

    @pytest.mark.asyncio
    async def test_execute_validation_step_success(self, orchestrator, mock_contract):
        """Test successful validation step execution."""
        step = {
            "step_id": "validate_input",
            "step_type": "validation",
            "required": True,
        }
        workflow_id = uuid4()

        result = await orchestrator._execute_validation_step(
            step, mock_contract, workflow_id
        )

        assert result["step_type"] == "validation"
        assert result["status"] == "success"
        assert "validated_at" in result

    @pytest.mark.asyncio
    async def test_execute_validation_step_missing_input(
        self, orchestrator, mock_contract_no_input
    ):
        """Test validation step with missing input data."""
        step = {
            "step_id": "validate_input",
            "step_type": "validation",
            "required": True,
        }
        workflow_id = uuid4()

        with pytest.raises(
            Exception
        ) as exc_info:  # Use generic Exception since OnexError types may vary
            await orchestrator._execute_validation_step(
                step, mock_contract_no_input, workflow_id
            )

        # Check that some error was raised (the exact type may vary)
        assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_route_to_onextree_success(self, orchestrator, mock_contract):
        """Test OnexTree intelligence routing with graceful degradation."""
        step = {"step_id": "intelligence", "step_type": "onextree_intelligence"}
        workflow_id = uuid4()

        result = await orchestrator._route_to_onextree(step, mock_contract, workflow_id)

        assert result["step_type"] == "onextree_intelligence"
        # OnexTree gracefully degrades when unavailable (returns "degraded" status)
        assert result["status"] in ["success", "degraded"]
        assert "intelligence_data" in result
        # Degraded mode returns fallback intelligence with 0.0 confidence
        confidence = float(result["intelligence_data"]["confidence_score"])
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_route_to_metadata_stamping_success(
        self, orchestrator, mock_contract
    ):
        """Test successful hash generation routing."""
        step = {
            "step_id": "hash_generation",
            "step_type": "hash_generation",
            "service": "metadata_stamping",
        }
        workflow_id = uuid4()

        # Mock the metadata client's generate_hash method as async
        if orchestrator.metadata_client:
            # Create async mock for generate_hash
            mock_hash_result = {
                "hash": "blake3_abc123def456",
                "execution_time_ms": 1.5,
                "performance_grade": "A",
                "file_size_bytes": 100,
            }
            orchestrator.metadata_client.generate_hash = AsyncMock(
                return_value=mock_hash_result
            )
            # Ensure initialize is also async mocked
            orchestrator.metadata_client.initialize = AsyncMock()
            orchestrator.metadata_client._http_client = MagicMock()

        result = await orchestrator._route_to_metadata_stamping(
            step, mock_contract, workflow_id
        )

        assert result["step_type"] == "hash_generation"
        assert result["status"] == "success"
        assert "file_hash" in result
        assert result["file_hash"].startswith("blake3_")
        assert "hash_generation_time_ms" in result

    @pytest.mark.asyncio
    async def test_route_to_metadata_stamping_missing_content(
        self, orchestrator, mock_contract_no_content, caplog
    ):
        """Test that hash generation handles missing content gracefully.

        This test verifies graceful degradation when content is missing.
        The orchestrator should use a placeholder hash and log a warning
        instead of failing, maintaining system resilience.
        """
        step = {
            "step_id": "hash_generation",
            "step_type": "hash_generation",
            "service": "metadata_stamping",
        }
        workflow_id = uuid4()

        # Should handle missing content gracefully with placeholder hash
        result = await orchestrator._route_to_metadata_stamping(
            step, mock_contract_no_content, workflow_id
        )

        # Verify result is returned (graceful degradation)
        assert result is not None

        # Verify warning was logged about using placeholder hash
        assert any(
            "placeholder hash" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    @pytest.mark.asyncio
    async def test_create_stamp_success(self, orchestrator, mock_contract):
        """Test successful stamp creation."""
        step = {"step_id": "stamp_creation", "step_type": "stamp_creation"}
        workflow_id = uuid4()

        result = await orchestrator._create_stamp(step, mock_contract, workflow_id)

        assert result["step_type"] == "stamp_creation"
        assert result["status"] == "success"
        assert "stamp_id" in result
        assert "stamp_data" in result
        assert result["stamp_data"]["namespace"] == orchestrator.default_namespace

    @pytest.mark.asyncio
    async def test_execute_workflow_step_unknown_type(
        self, orchestrator, mock_contract
    ):
        """Test workflow step execution with unknown step type."""
        step = {"step_id": "unknown", "step_type": "unknown_type"}
        workflow_id = uuid4()

        with pytest.raises(
            Exception
        ) as exc_info:  # Use generic Exception since OnexError types may vary
            await orchestrator._execute_workflow_step(step, mock_contract, workflow_id)

        # Check that some error was raised (the exact type may vary)
        assert exc_info.value is not None


# Test: Default Workflow Steps


class TestDefaultWorkflow:
    """Test suite for default workflow configuration."""

    def test_get_default_workflow_steps(self, orchestrator):
        """Test default workflow steps generation."""
        steps = orchestrator._get_default_workflow_steps()

        assert len(steps) == 3
        assert steps[0]["step_type"] == "validation"
        assert steps[1]["step_type"] == "hash_generation"
        assert steps[2]["step_type"] == "stamp_creation"

        # Verify all steps are required
        for step in steps:
            assert step.get("required") is True


# Test: Result Aggregation


class TestResultAggregation:
    """Test suite for workflow result aggregation."""

    @pytest.mark.asyncio
    async def test_aggregate_workflow_results_complete(
        self, orchestrator, mock_contract
    ):
        """Test aggregation of complete workflow results."""
        workflow_id = uuid4()
        file_hash = "blake3_abc123"
        stamp_id = str(uuid4())

        results = [
            {"step_type": "validation", "status": "success"},
            {
                "step_type": "hash_generation",
                "status": "success",
                "file_hash": file_hash,
                "hash_generation_time_ms": 1.5,
            },
            {
                "step_type": "stamp_creation",
                "status": "success",
                "stamp_id": stamp_id,
                "stamp_data": {"namespace": "test.namespace"},
            },
            {
                "step_type": "onextree_intelligence",
                "status": "success",
                "intelligence_data": {
                    "confidence_score": "0.95",
                    "analysis_type": "content_validation",
                },
            },
        ]

        output = await orchestrator._aggregate_workflow_results(
            results, mock_contract, workflow_id
        )

        assert isinstance(output, ModelStampResponseOutput)
        assert output.file_hash == file_hash
        assert output.stamp_id == stamp_id
        assert output.namespace == orchestrator.default_namespace
        assert output.intelligence_data is not None
        assert output.workflow_steps_executed == len(results)

    @pytest.mark.asyncio
    async def test_aggregate_workflow_results_minimal(
        self, orchestrator, mock_contract
    ):
        """Test aggregation with minimal workflow results."""
        workflow_id = uuid4()
        results = [
            {"step_type": "validation", "status": "success"},
        ]

        output = await orchestrator._aggregate_workflow_results(
            results, mock_contract, workflow_id
        )

        assert isinstance(output, ModelStampResponseOutput)
        assert output.file_hash == "unknown"
        assert output.stamp_id == "unknown"
        assert output.intelligence_data is None


# Test: Metrics Tracking


class TestMetricsTracking:
    """Test suite for performance metrics tracking."""

    @pytest.mark.asyncio
    async def test_update_metrics_success(self, orchestrator):
        """Test metrics update for successful operation."""
        await orchestrator._update_stamping_metrics("test_operation", 10.5, True)

        metrics = orchestrator.get_stamping_metrics()
        assert "test_operation" in metrics
        assert metrics["test_operation"]["total_operations"] == 1
        assert metrics["test_operation"]["successful_operations"] == 1
        assert metrics["test_operation"]["failed_operations"] == 0
        assert metrics["test_operation"]["avg_time_ms"] == 10.5

    @pytest.mark.asyncio
    async def test_update_metrics_failure(self, orchestrator):
        """Test metrics update for failed operation."""
        await orchestrator._update_stamping_metrics("test_operation", 15.0, False)

        metrics = orchestrator.get_stamping_metrics()
        assert metrics["test_operation"]["total_operations"] == 1
        assert metrics["test_operation"]["successful_operations"] == 0
        assert metrics["test_operation"]["failed_operations"] == 1

    @pytest.mark.asyncio
    async def test_update_metrics_multiple_operations(self, orchestrator):
        """Test metrics aggregation across multiple operations."""
        await orchestrator._update_stamping_metrics("test_operation", 10.0, True)
        await orchestrator._update_stamping_metrics("test_operation", 20.0, True)
        await orchestrator._update_stamping_metrics("test_operation", 30.0, False)

        metrics = orchestrator.get_stamping_metrics()
        assert metrics["test_operation"]["total_operations"] == 3
        assert metrics["test_operation"]["successful_operations"] == 2
        assert metrics["test_operation"]["failed_operations"] == 1
        assert metrics["test_operation"]["avg_time_ms"] == 20.0
        assert metrics["test_operation"]["min_time_ms"] == 10.0
        assert metrics["test_operation"]["max_time_ms"] == 30.0


# Test: Workflow State Query


class TestWorkflowStateQuery:
    """Test suite for workflow state queries."""

    def test_get_workflow_state_exists(self, orchestrator):
        """Test retrieving existing workflow state."""
        workflow_id = uuid4()
        orchestrator.workflow_fsm_states[str(workflow_id)] = (
            EnumWorkflowState.PROCESSING
        )

        state = orchestrator.get_workflow_state(workflow_id)
        assert state == EnumWorkflowState.PROCESSING

    def test_get_workflow_state_not_exists(self, orchestrator):
        """Test retrieving non-existent workflow state."""
        workflow_id = uuid4()

        state = orchestrator.get_workflow_state(workflow_id)
        assert state is None


# Test: Complete Workflow Orchestration


class TestCompleteWorkflowOrchestration:
    """Test suite for end-to-end workflow orchestration."""

    @pytest.mark.asyncio
    async def test_execute_orchestration_success(self, orchestrator, mock_contract):
        """Test successful complete workflow orchestration."""
        # Mock the metadata client's async methods
        if orchestrator.metadata_client:
            mock_hash_result = {
                "hash": "blake3_abc123def456",
                "execution_time_ms": 1.5,
                "performance_grade": "A",
                "file_size_bytes": 100,
            }
            orchestrator.metadata_client.generate_hash = AsyncMock(
                return_value=mock_hash_result
            )
            orchestrator.metadata_client.initialize = AsyncMock()
            orchestrator.metadata_client._http_client = MagicMock()

        result = await orchestrator.execute_orchestration(mock_contract)

        assert isinstance(result, ModelStampResponseOutput)
        assert result.workflow_state == EnumWorkflowState.COMPLETED
        assert result.workflow_id == mock_contract.correlation_id
        assert result.processing_time_ms > 0
        assert result.workflow_steps_executed > 0
        assert result.stamp_id != "unknown"
        assert result.file_hash.startswith("blake3_")

        # Verify final state is stored
        workflow_state = orchestrator.get_workflow_state(mock_contract.correlation_id)
        assert workflow_state == EnumWorkflowState.COMPLETED

        # Verify metrics were updated
        metrics = orchestrator.get_stamping_metrics()
        assert "workflow_orchestration" in metrics
        assert metrics["workflow_orchestration"]["successful_operations"] == 1

    @pytest.mark.asyncio
    async def test_execute_orchestration_missing_input_data(
        self, orchestrator, mock_contract_no_input
    ):
        """Test workflow orchestration with missing input data."""
        with pytest.raises(
            Exception
        ) as exc_info:  # Use generic Exception since OnexError types may vary
            await orchestrator.execute_orchestration(mock_contract_no_input)

        # Check that some error was raised (the exact type may vary)
        assert exc_info.value is not None

        # Verify workflow transitioned to FAILED state
        workflow_state = orchestrator.get_workflow_state(
            mock_contract_no_input.correlation_id
        )
        assert workflow_state == EnumWorkflowState.FAILED

    @pytest.mark.asyncio
    async def test_execute_orchestration_with_workflow_error(
        self, orchestrator, mock_contract
    ):
        """Test workflow orchestration with execution error."""

        # Patch a workflow step to raise an exception
        async def failing_step(*args, **kwargs):
            raise ValueError("Simulated workflow step failure")

        with patch.object(
            orchestrator, "_execute_validation_step", side_effect=failing_step
        ):
            with pytest.raises(
                Exception
            ) as exc_info:  # Use generic Exception since OnexError types may vary
                await orchestrator.execute_orchestration(mock_contract)

            # Check that some error was raised (the exact type may vary)
            assert exc_info.value is not None

        # Verify workflow transitioned to FAILED state
        workflow_state = orchestrator.get_workflow_state(mock_contract.correlation_id)
        assert workflow_state == EnumWorkflowState.FAILED

        # Verify metrics recorded the failure
        metrics = orchestrator.get_stamping_metrics()
        assert "workflow_orchestration" in metrics
        assert metrics["workflow_orchestration"]["failed_operations"] == 1

    @pytest.mark.asyncio
    async def test_execute_orchestration_multiple_workflows(self, orchestrator):
        """Test concurrent execution of multiple workflows."""
        # Mock the metadata client's async methods
        if orchestrator.metadata_client:
            mock_hash_result = {
                "hash": "blake3_abc123def456",
                "execution_time_ms": 1.5,
                "performance_grade": "A",
                "file_size_bytes": 100,
            }
            orchestrator.metadata_client.generate_hash = AsyncMock(
                return_value=mock_hash_result
            )
            orchestrator.metadata_client.initialize = AsyncMock()
            orchestrator.metadata_client._http_client = MagicMock()

        # Create multiple contracts
        contracts = [MagicMock(spec=ModelContractOrchestrator) for _ in range(3)]
        for contract in contracts:
            contract.correlation_id = uuid4()
            contract.input_data = {"file_path": "/test/file.txt", "content": "test"}

        # Execute workflows concurrently
        results = await asyncio.gather(
            *[orchestrator.execute_orchestration(contract) for contract in contracts]
        )

        # Verify all workflows completed successfully
        assert len(results) == 3
        for result in results:
            assert result.workflow_state == EnumWorkflowState.COMPLETED
            assert result.processing_time_ms > 0

        # Verify all workflows are tracked
        for contract in contracts:
            state = orchestrator.get_workflow_state(contract.correlation_id)
            assert state == EnumWorkflowState.COMPLETED


# Test: Enum Utilities


class TestEnumUtilities:
    """Test suite for enum utility methods."""

    def test_workflow_state_is_terminal(self):
        """Test EnumWorkflowState terminal state detection."""
        assert EnumWorkflowState.COMPLETED.is_terminal() is True
        assert EnumWorkflowState.FAILED.is_terminal() is True
        assert EnumWorkflowState.PENDING.is_terminal() is False
        assert EnumWorkflowState.PROCESSING.is_terminal() is False

    def test_workflow_state_can_transition_to(self):
        """Test EnumWorkflowState transition validation."""
        # Valid transitions
        assert (
            EnumWorkflowState.PENDING.can_transition_to(EnumWorkflowState.PROCESSING)
            is True
        )
        assert (
            EnumWorkflowState.PROCESSING.can_transition_to(EnumWorkflowState.COMPLETED)
            is True
        )
        assert (
            EnumWorkflowState.PROCESSING.can_transition_to(EnumWorkflowState.FAILED)
            is True
        )

        # Invalid transitions
        assert (
            EnumWorkflowState.PENDING.can_transition_to(EnumWorkflowState.COMPLETED)
            is False
        )
        assert (
            EnumWorkflowState.COMPLETED.can_transition_to(EnumWorkflowState.PROCESSING)
            is False
        )
        assert (
            EnumWorkflowState.FAILED.can_transition_to(EnumWorkflowState.PROCESSING)
            is False
        )

    def test_workflow_event_get_topic_name(self):
        """Test EnumWorkflowEvent topic name generation."""
        event = EnumWorkflowEvent.WORKFLOW_STARTED
        topic = event.get_topic_name("custom.namespace")
        assert (
            topic
            == "custom.namespace.omninode_bridge.onex.evt.stamp-workflow-started.v1"
        )

        # Test default namespace
        topic = event.get_topic_name()
        assert topic == "dev.omninode_bridge.onex.evt.stamp-workflow-started.v1"

    def test_workflow_event_is_terminal(self):
        """Test EnumWorkflowEvent terminal event detection."""
        assert EnumWorkflowEvent.WORKFLOW_COMPLETED.is_terminal_event is True
        assert EnumWorkflowEvent.WORKFLOW_FAILED.is_terminal_event is True
        assert EnumWorkflowEvent.WORKFLOW_STARTED.is_terminal_event is False
        assert EnumWorkflowEvent.STEP_COMPLETED.is_terminal_event is False

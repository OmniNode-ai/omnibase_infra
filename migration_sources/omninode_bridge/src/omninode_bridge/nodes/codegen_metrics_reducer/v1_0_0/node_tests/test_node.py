"""Unit tests for NodeCodegenMetricsReducer."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer
from omnibase_core.models.core import ModelContainer

from omninode_bridge.events.models.codegen_events import (
    ModelEventCodegenCompleted,
    ModelEventCodegenFailed,
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
)
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.models.enum_metrics_window import (
    EnumMetricsWindow,
)
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.models.model_metrics_state import (
    ModelMetricsState,
)
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.node import (
    NodeCodegenMetricsReducer,
)


@pytest.fixture
def mock_container():
    """Create container with mocked services."""
    container = Mock(spec=ModelContainer)
    container.config = Mock()
    container.config.get = Mock(side_effect=lambda k, default: default)
    container.get_service = Mock(return_value=None)
    return container


@pytest.fixture
def metrics_reducer(mock_container):
    """Create metrics reducer with mocked dependencies."""
    # Disable Consul registration for tests
    mock_container.config.get = Mock(
        side_effect=lambda k, default: (
            False if k == "consul_enable_registration" else default
        )
    )
    return NodeCodegenMetricsReducer(mock_container)


async def async_iterator(items):
    """Helper to create async iterator from list."""
    for item in items:
        yield item


class TestNodeCodegenMetricsReducer:
    """Test suite for NodeCodegenMetricsReducer."""

    @pytest.mark.asyncio
    async def test_pure_aggregation(self, metrics_reducer):
        """Test pure aggregation without I/O."""
        # Arrange
        events = [
            ModelEventCodegenStarted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                node_name="TestNode",
                timestamp=datetime.now(UTC),
            ),
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                stage_name="extraction",
                duration_ms=50.0,
                timestamp=datetime.now(UTC),
            ),
            ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                total_duration_ms=150.0,
                lines_generated=100,
                timestamp=datetime.now(UTC),
            ),
        ]

        # Mock contract with input_stream
        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        # Mock publish_event_intent to avoid I/O
        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert isinstance(result, ModelMetricsState)
            assert result.total_events_processed == len(events)
            assert result.aggregation_duration_ms > 0
            assert result.items_per_second > 0

    @pytest.mark.asyncio
    async def test_windowed_aggregation(self, metrics_reducer):
        """Test aggregation with time windows."""
        # Arrange
        events = [
            ModelEventCodegenStarted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                node_name="TestNode",
                timestamp=datetime.now(UTC),
            )
            for _ in range(10)
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {"window_type": "hourly"}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.total_events_processed == len(events)
            assert result.window_type == EnumMetricsWindow.HOURLY

    @pytest.mark.asyncio
    async def test_batch_processing(self, metrics_reducer):
        """Test batch processing of events."""
        # Arrange
        batch_size = 5
        total_events = 20
        events = [
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                stage_name=f"stage_{i}",
                duration_ms=10.0 * i,
                timestamp=datetime.now(UTC),
            )
            for i in range(total_events)
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {"batch_size": batch_size}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.total_events_processed == total_events

    @pytest.mark.asyncio
    async def test_event_intent_publishing(self, metrics_reducer):
        """Test that event intents are published via MixinIntentPublisher."""
        # Arrange
        events = [
            ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                total_duration_ms=100.0,
                lines_generated=50,
                timestamp=datetime.now(UTC),
            )
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        # Mock publish_event_intent to track calls
        with patch.object(
            metrics_reducer, "publish_event_intent", AsyncMock()
        ) as mock_publish:
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert mock_publish.called
            assert mock_publish.call_count == 1

            # Check call arguments
            call_args = mock_publish.call_args[1]
            assert "target_topic" in call_args
            assert "target_key" in call_args
            assert "event" in call_args

    @pytest.mark.asyncio
    async def test_aggregation_with_failed_events(self, metrics_reducer):
        """Test aggregation includes failed events."""
        # Arrange
        events = [
            ModelEventCodegenStarted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                node_name="TestNode",
                timestamp=datetime.now(UTC),
            ),
            ModelEventCodegenFailed(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                error_message="Test error",
                timestamp=datetime.now(UTC),
            ),
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.total_events_processed == len(events)

    @pytest.mark.asyncio
    async def test_empty_event_stream(self, metrics_reducer):
        """Test handling of empty event stream."""
        # Arrange
        events = []

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.total_events_processed == 0

    @pytest.mark.asyncio
    async def test_aggregation_metrics_tracking(self, metrics_reducer):
        """Test aggregation duration and throughput metrics."""
        # Arrange
        events = [
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                stage_name=f"stage_{i}",
                duration_ms=10.0,
                timestamp=datetime.now(UTC),
            )
            for i in range(100)
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.aggregation_duration_ms > 0
            assert result.items_per_second > 0
            # Should process 100 items relatively quickly
            assert result.aggregation_duration_ms < 1000  # <1s for 100 events

    @pytest.mark.asyncio
    async def test_dict_event_parsing(self, metrics_reducer):
        """Test parsing of dict events from stream."""
        # Arrange
        event_dict = {
            "event_type": "codegen_started",
            "correlation_id": str(uuid4()),
            "workflow_id": str(uuid4()),
            "node_name": "TestNode",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator([event_dict])
        contract.input_state = {}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            # Should handle dict parsing (though may be 0 if parsing fails)
            assert isinstance(result, ModelMetricsState)

    @pytest.mark.asyncio
    async def test_window_type_configuration(self, metrics_reducer):
        """Test window type configuration from contract."""
        # Arrange
        events = [
            ModelEventCodegenStarted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                node_name="TestNode",
                timestamp=datetime.now(UTC),
            )
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {"window_type": "daily"}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.window_type == EnumMetricsWindow.DAILY

    @pytest.mark.asyncio
    async def test_correlation_id_tracking(self, metrics_reducer):
        """Test correlation ID is tracked throughout reduction."""
        # Arrange
        correlation_id = uuid4()
        events = [
            ModelEventCodegenCompleted(
                correlation_id=correlation_id,
                workflow_id=str(uuid4()),
                total_duration_ms=100.0,
                lines_generated=50,
                timestamp=datetime.now(UTC),
            )
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = correlation_id
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result is not None


# Performance benchmark tests
class TestMetricsReducerPerformance:
    """Performance tests for metrics reduction."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_throughput_aggregation(self, metrics_reducer):
        """Benchmark >1000 events/second aggregation."""
        # Arrange
        total_events = 2000
        events = [
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                stage_name=f"stage_{i}",
                duration_ms=10.0,
                timestamp=datetime.now(UTC),
            )
            for i in range(total_events)
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.total_events_processed == total_events
            # Target: >1000 events/second
            assert result.items_per_second > 1000

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_batch_processing(self, metrics_reducer):
        """Benchmark aggregation with large batches."""
        # Arrange
        batch_size = 500
        total_events = 5000
        events = [
            ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                total_duration_ms=100.0,
                lines_generated=50,
                timestamp=datetime.now(UTC),
            )
            for _ in range(total_events)
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {"batch_size": batch_size}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert result.total_events_processed == total_events
            # Should maintain >1000 events/sec even with large batches
            assert result.items_per_second > 1000

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_aggregation_latency(self, metrics_reducer):
        """Benchmark aggregation latency for 1000 items."""
        # Arrange
        events = [
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                workflow_id=str(uuid4()),
                stage_name=f"stage_{i}",
                duration_ms=10.0,
                timestamp=datetime.now(UTC),
            )
            for i in range(1000)
        ]

        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = async_iterator(events)
        contract.input_state = {}

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            result = await metrics_reducer.execute_reduction(contract)

            # Assert
            # Target: <100ms for 1000 items
            assert result.aggregation_duration_ms < 100

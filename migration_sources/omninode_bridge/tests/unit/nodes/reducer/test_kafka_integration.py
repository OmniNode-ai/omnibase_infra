#!/usr/bin/env python3
"""Unit tests for NodeBridgeReducer Kafka integration.

Comprehensive test coverage for:
- Kafka client initialization
- Event publishing at lifecycle points
- Error handling and failure events
- Event payload validation
- OnexEnvelopeV1 wrapping
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import with fallback to stubs when omnibase_core is not available
try:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.contracts.model_contract_base import (
        EnumNodeType,
        ModelSemVer,
    )
    from omnibase_core.models.contracts.model_contract_reducer import (
        ModelContractReducer,
    )
except ImportError:
    # Fallback to stub implementations
    from omninode_bridge.nodes.reducer.v1_0_0._stubs import (
        ModelContractReducer,
        ModelONEXContainer,
    )

    # These enums need stub definitions too
    class EnumNodeType:
        REDUCER = "reducer"

    class ModelSemVer:
        def __init__(self, major: int, minor: int, patch: int):
            self.major = major
            self.minor = minor
            self.patch = patch


from omninode_bridge.nodes.reducer.v1_0_0.models.enum_reducer_event import (
    EnumReducerEvent,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_input_state import (
    ModelReducerInputState,
)
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer


@pytest.fixture
def mock_kafka_client() -> MagicMock:
    """Create a mock Kafka client."""
    client = MagicMock()
    client.is_connected = True
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.publish_with_envelope = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_container_with_kafka(
    mock_kafka_client: MagicMock,
) -> MagicMock:
    """Create a mock container with Kafka client."""
    container = MagicMock(spec=ModelONEXContainer)
    # Set up container.config as a dict-like mock that returns test values
    container.config = {
        "default_namespace": "test",
        "kafka_broker_url": "localhost:9092",
    }
    # Mock container.value to return the same config (for fallback code path)
    container.value = container.config
    # Mock get_service to return the mock Kafka client
    container.get_service = MagicMock(return_value=mock_kafka_client)
    # Mock register_service (no-op for tests)
    container.register_service = MagicMock(return_value=None)
    return container


@pytest.fixture
def reducer_with_kafka(
    mock_container_with_kafka: ModelONEXContainer,
) -> NodeBridgeReducer:
    """Create NodeBridgeReducer with Kafka client."""
    # Patch get_reducer_config to force fallback to container config
    with patch(
        "omninode_bridge.config.config_loader.get_reducer_config",
        side_effect=ImportError("Mocked ConfigLoader failure for testing"),
    ):
        node = NodeBridgeReducer(container=mock_container_with_kafka)
    return node


@pytest.fixture
def sample_metadata() -> ModelReducerInputState:
    """Create sample metadata for testing."""
    return ModelReducerInputState(
        stamp_id=str(uuid4()),
        file_hash="abc123",
        file_path="/data/test/sample.txt",
        file_size=1024,
        workflow_id=uuid4(),
        namespace="test_namespace",
        content_type="text/plain",
        workflow_state="COMPLETED",
    )


class TestKafkaClientInitialization:
    """Test Kafka client initialization in reducer."""

    def test_kafka_client_from_container(
        self,
        mock_container_with_kafka: ModelONEXContainer,
        mock_kafka_client: MagicMock,
    ) -> None:
        """Test that Kafka client is retrieved from container."""
        node = NodeBridgeReducer(container=mock_container_with_kafka)
        assert node.kafka_client is mock_kafka_client
        assert node.kafka_client.is_connected

    def test_kafka_client_none_in_health_check_mode(self) -> None:
        """Test that Kafka client is None when health check mode is enabled."""
        container = MagicMock(spec=ModelONEXContainer)
        container.config = MagicMock()
        container.config.get = MagicMock(
            side_effect=lambda k, default=None: {
                "default_namespace": "test",
                "health_check_mode": True,
            }.get(k, default)
        )
        container.get_service = MagicMock(return_value=None)
        node = NodeBridgeReducer(container=container)
        assert node.kafka_client is None

    @patch("omninode_bridge.services.kafka_client.KafkaClient")
    def test_kafka_client_creation_when_not_in_container(
        self,
        mock_kafka_class: MagicMock,
    ) -> None:
        """Test Kafka client creation when not in container."""
        mock_client = MagicMock()
        mock_kafka_class.return_value = mock_client

        container = MagicMock(spec=ModelONEXContainer)
        container.config = MagicMock()
        container.config.get = MagicMock(
            side_effect=lambda k, default=None: {
                "default_namespace": "test",
                "kafka_broker_url": "localhost:9092",
            }.get(k, default)
        )
        container.get_service = MagicMock(
            return_value=None
        )  # No kafka_client in container

        node = NodeBridgeReducer(container=container)

        # Verify KafkaClient was created
        mock_kafka_class.assert_called_once_with(
            bootstrap_servers="localhost:9092",
            enable_dead_letter_queue=True,
            max_retry_attempts=3,
            timeout_seconds=30,
        )

        # Verify client is available in node
        assert node.kafka_client is mock_client


class TestEventPublishing:
    """Test event publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_aggregation_started_event(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test AGGREGATION_STARTED event is published."""
        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        await reducer_with_kafka.execute_reduction(contract)

        # Verify AGGREGATION_STARTED was published
        calls = mock_kafka_client.publish_with_envelope.call_args_list
        started_calls = [
            c for c in calls if c.kwargs.get("event_type") == "aggregation_started"
        ]

        assert len(started_calls) == 1
        assert (
            started_calls[0].kwargs["topic"]
            == "test.omninode_bridge.onex.evt.aggregation-started.v1"
        )

    @pytest.mark.asyncio
    async def test_publish_batch_processed_event(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
    ) -> None:
        """Test BATCH_PROCESSED event is published."""
        metadata_batch = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash=f"hash{i}",
                file_path=f"/data/test/sample{i}.txt",
                file_size=1024,
                workflow_id=uuid4(),
                namespace="test",
                content_type="text/plain",
                workflow_state="COMPLETED",
            )
            for i in range(5)
        ]

        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [m.model_dump() for m in metadata_batch]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        await reducer_with_kafka.execute_reduction(contract)

        # Verify BATCH_PROCESSED was published
        calls = mock_kafka_client.publish_with_envelope.call_args_list
        batch_calls = [
            c for c in calls if c.kwargs.get("event_type") == "batch_processed"
        ]

        assert len(batch_calls) >= 1  # At least one batch

    @pytest.mark.asyncio
    async def test_publish_aggregation_completed_event(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test AGGREGATION_COMPLETED event is published."""
        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        await reducer_with_kafka.execute_reduction(contract)

        # Verify AGGREGATION_COMPLETED was published
        calls = mock_kafka_client.publish_with_envelope.call_args_list
        completed_calls = [
            c for c in calls if c.kwargs.get("event_type") == "aggregation_completed"
        ]

        assert len(completed_calls) == 1
        assert (
            completed_calls[0].kwargs["topic"]
            == "test.omninode_bridge.onex.evt.aggregation-completed.v1"
        )

    @pytest.mark.asyncio
    async def test_publish_aggregation_failed_event_on_error(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
    ) -> None:
        """Test AGGREGATION_FAILED event is published on error."""
        # Create invalid contract to trigger error
        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state=None,  # Invalid - will cause error
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        with pytest.raises(Exception):
            await reducer_with_kafka.execute_reduction(contract)

        # Verify AGGREGATION_FAILED was published
        calls = mock_kafka_client.publish_with_envelope.call_args_list
        failed_calls = [
            c for c in calls if c.kwargs.get("event_type") == "aggregation_failed"
        ]

        assert len(failed_calls) == 1
        failed_payload = failed_calls[0].kwargs["payload"]
        assert "error_type" in failed_payload
        assert "error_message" in failed_payload


class TestEventPayloads:
    """Test event payload structure and content."""

    @pytest.mark.asyncio
    async def test_aggregation_started_payload_structure(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test AGGREGATION_STARTED event payload structure."""
        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        await reducer_with_kafka.execute_reduction(contract)

        # Get AGGREGATION_STARTED call
        calls = mock_kafka_client.publish_with_envelope.call_args_list
        started_call = next(
            c for c in calls if c.kwargs.get("event_type") == "aggregation_started"
        )

        payload = started_call.kwargs["payload"]

        # Verify payload structure
        assert "aggregation_id" in payload
        assert "aggregation_type" in payload
        assert "batch_size" in payload
        assert "window_size_ms" in payload
        assert "timestamp" in payload
        assert "node_id" in payload
        assert "published_at" in payload

    @pytest.mark.asyncio
    async def test_aggregation_completed_payload_metrics(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test AGGREGATION_COMPLETED event includes performance metrics."""
        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        await reducer_with_kafka.execute_reduction(contract)

        # Get AGGREGATION_COMPLETED call
        calls = mock_kafka_client.publish_with_envelope.call_args_list
        completed_call = next(
            c for c in calls if c.kwargs.get("event_type") == "aggregation_completed"
        )

        payload = completed_call.kwargs["payload"]

        # Verify performance metrics
        assert "duration_ms" in payload
        assert "items_per_second" in payload
        assert "total_items" in payload
        assert "batches_processed" in payload
        assert "total_size_bytes" in payload
        assert "namespaces_count" in payload

        assert payload["duration_ms"] > 0
        assert payload["items_per_second"] > 0


class TestOnexEnvelopeWrapping:
    """Test OnexEnvelopeV1 wrapping."""

    @pytest.mark.asyncio
    async def test_events_use_onex_envelope(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test that events are wrapped in OnexEnvelopeV1."""
        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        await reducer_with_kafka.execute_reduction(contract)

        # Verify publish_with_envelope was called (not raw publish)
        assert mock_kafka_client.publish_with_envelope.called
        assert (
            mock_kafka_client.publish_with_envelope.call_count >= 3
        )  # Started, batch, completed

    @pytest.mark.asyncio
    async def test_envelope_metadata_includes_node_info(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test envelope metadata includes node information."""
        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        await reducer_with_kafka.execute_reduction(contract)

        # Check metadata in first call
        first_call = mock_kafka_client.publish_with_envelope.call_args_list[0]
        metadata = first_call.kwargs["metadata"]

        assert metadata["event_category"] == "metadata_aggregation"
        assert metadata["node_type"] == "reducer"
        assert metadata["namespace"] == "test"


class TestLifecycleHooks:
    """Test startup and shutdown lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_startup_connects_kafka(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
    ) -> None:
        """Test that startup connects Kafka client."""
        mock_kafka_client.is_connected = False

        await reducer_with_kafka.startup()

        # Verify Kafka client was connected
        mock_kafka_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_kafka(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
    ) -> None:
        """Test that shutdown disconnects Kafka client."""
        await reducer_with_kafka.shutdown()

        # Verify Kafka client was disconnected
        mock_kafka_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_handles_connection_failure_gracefully(
        self,
        reducer_with_kafka: NodeBridgeReducer,
        mock_kafka_client: MagicMock,
    ) -> None:
        """Test that startup handles Kafka connection failures gracefully."""
        mock_kafka_client.is_connected = False
        mock_kafka_client.connect.side_effect = Exception("Connection failed")

        # Should not raise exception
        await reducer_with_kafka.startup()


class TestKafkaClientFallback:
    """Test behavior when Kafka client is unavailable."""

    @pytest.mark.asyncio
    async def test_execute_reduction_without_kafka_client(
        self,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test execution works without Kafka client."""
        container = ModelONEXContainer()
        # Don't register kafka_client to simulate health check mode
        node = NodeBridgeReducer(container=container)

        contract = ModelContractReducer(
            name="test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        # Should complete without errors
        result = await node.execute_reduction(contract)

        assert result.total_items == 1

    @pytest.mark.asyncio
    async def test_publish_event_logs_when_kafka_unavailable(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that events are logged when Kafka is unavailable."""
        import logging

        # Explicitly set caplog to capture INFO level
        caplog.set_level(logging.INFO)

        container = ModelONEXContainer()
        # Don't register kafka_client to simulate health check mode
        node = NodeBridgeReducer(container=container)

        # Call _publish_event directly
        await node._publish_event(
            EnumReducerEvent.AGGREGATION_STARTED,
            {"aggregation_id": "test123"},
        )

        # Verify event was logged (not published to Kafka)
        assert (
            "Kafka event (no client)" in caplog.text
            or "aggregation_started" in caplog.text
        )

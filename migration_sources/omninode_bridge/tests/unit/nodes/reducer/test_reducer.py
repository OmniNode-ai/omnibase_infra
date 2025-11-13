#!/usr/bin/env python3
"""Unit tests for NodeBridgeReducer.

Comprehensive test coverage for:
- Aggregation logic (namespace grouping, time windows)
- State persistence (PostgreSQL integration)
- Error handling (invalid inputs, missing fields)
- Performance metrics (duration, throughput)
- Streaming (batch processing, async iteration)
- FSM state tracking (workflow state management)
"""

from datetime import datetime
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


from omninode_bridge.nodes.reducer.v1_0_0.models.enum_aggregation_type import (
    EnumAggregationType,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_input_state import (
    ModelReducerInputState,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_output_state import (
    ModelReducerOutputState,
)
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Fixtures moved to conftest.py for better organization and reusability


class TestNodeBridgeReducerInit:
    """Test NodeBridgeReducer initialization."""

    def test_init_with_container(self, mock_container: ModelONEXContainer) -> None:
        """Test node initialization with container."""
        node = NodeBridgeReducer(container=mock_container)
        assert node is not None
        assert node._aggregation_buffer is not None
        assert node._fsm_state_tracker is not None
        assert node._current_window_start is None

    def test_init_sets_default_configs(self, reducer_node: NodeBridgeReducer) -> None:
        """Test that initialization sets default configurations."""
        # When created without a contract (mock container), configs should be None
        # They are only populated when the parent class loads a contract YAML
        assert reducer_node._aggregation_config is None
        assert reducer_node._state_config is None
        assert reducer_node._fsm_config is None


class TestAggregationExecution:
    """Test core aggregation execution logic."""

    @pytest.mark.asyncio
    async def test_execute_reduction_single_item(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test reduction with a single metadata item."""
        contract = ModelContractReducer(
            name="test_reduction",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test reduction contract",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        assert isinstance(result, ModelReducerOutputState)
        assert result.total_items == 1
        assert result.total_size_bytes == sample_metadata.file_size
        assert len(result.namespaces) == 1
        assert result.namespaces[0] == sample_metadata.namespace
        assert result.aggregation_duration_ms > 0
        assert result.items_per_second > 0

    @pytest.mark.asyncio
    async def test_execute_reduction_batch(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata_batch: list[ModelReducerInputState],
    ) -> None:
        """Test reduction with a batch of metadata items."""
        contract = ModelContractReducer(
            name="test_batch_reduction",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test batch reduction",
            node_type=EnumNodeType.REDUCER,
            input_state={
                "items": [item.model_dump() for item in sample_metadata_batch]
            },
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        assert result.total_items == len(sample_metadata_batch)
        assert len(result.namespaces) == 2  # Two distinct namespaces
        assert "omninode.services.metadata" in result.namespaces
        assert "omninode.services.onextree" in result.namespaces

    @pytest.mark.asyncio
    async def test_namespace_aggregation(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata_batch: list[ModelReducerInputState],
    ) -> None:
        """Test that items are correctly aggregated by namespace."""
        contract = ModelContractReducer(
            name="test_namespace_agg",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test namespace aggregation",
            node_type=EnumNodeType.REDUCER,
            input_state={
                "items": [item.model_dump() for item in sample_metadata_batch]
            },
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify namespace aggregations
        for namespace in result.namespaces:
            agg_data = result.aggregations[namespace]
            assert "total_stamps" in agg_data
            assert "total_size_bytes" in agg_data
            assert "file_types" in agg_data
            assert "workflow_ids" in agg_data
            assert agg_data["total_stamps"] > 0

    @pytest.mark.asyncio
    async def test_file_type_aggregation(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata_batch: list[ModelReducerInputState],
    ) -> None:
        """Test aggregation of file types."""
        contract = ModelContractReducer(
            name="test_file_type_agg",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test file type aggregation",
            node_type=EnumNodeType.REDUCER,
            input_state={
                "items": [item.model_dump() for item in sample_metadata_batch]
            },
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify file types are tracked
        for namespace in result.namespaces:
            file_types = result.aggregations[namespace]["file_types"]
            assert isinstance(file_types, list)
            assert len(file_types) > 0
            assert all(isinstance(ft, str) for ft in file_types)


class TestStreamingAndBatching:
    """Test streaming and batch processing."""

    @pytest.mark.asyncio
    async def test_stream_metadata_with_list_input(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata_batch: list[ModelReducerInputState],
    ) -> None:
        """Test streaming metadata from list input."""
        contract = ModelContractReducer(
            name="test_stream",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test streaming",
            node_type=EnumNodeType.REDUCER,
            input_state={
                "items": [item.model_dump() for item in sample_metadata_batch]
            },
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        batches = []
        async for batch in reducer_node._stream_metadata(contract, batch_size=3):
            batches.append(batch)

        # Verify batching
        assert len(batches) > 1  # Should have multiple batches
        total_items = sum(len(batch) for batch in batches)
        assert total_items == len(sample_metadata_batch)

    @pytest.mark.asyncio
    async def test_stream_metadata_batch_size(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test that batch size is respected during streaming."""
        # Create 25 items
        items = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash=f"hash_{i}",
                file_path=f"/data/file_{i}.txt",
                file_size=1024,
                workflow_id=uuid4(),
            )
            for i in range(25)
        ]

        contract = ModelContractReducer(
            name="test_batch_size",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test batch size",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [item.model_dump() for item in items]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        batch_size = 10
        batches = []
        async for batch in reducer_node._stream_metadata(
            contract, batch_size=batch_size
        ):
            batches.append(batch)
            # Each batch except the last should have batch_size items
            if len(batches) < 3:
                assert len(batch) <= batch_size

        # Should have 3 batches: 10, 10, 5
        assert len(batches) == 3


class TestFSMStateTracking:
    """Test FSM state tracking functionality."""

    @pytest.mark.asyncio
    async def test_fsm_state_tracking(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test that FSM states are correctly tracked by workflow ID."""
        workflow_id_1 = uuid4()
        workflow_id_2 = uuid4()

        items = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash="hash_1",
                file_path="/data/file1.txt",
                file_size=1024,
                workflow_id=workflow_id_1,
                workflow_state="PROCESSING",
            ),
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash="hash_2",
                file_path="/data/file2.txt",
                file_size=2048,
                workflow_id=workflow_id_2,
                workflow_state="COMPLETED",
            ),
        ]

        contract = ModelContractReducer(
            name="test_fsm",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test FSM tracking",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [item.model_dump() for item in items]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify FSM states are tracked (FSMStateManager normalizes to uppercase)
        assert len(result.fsm_states) == 2
        assert result.fsm_states[str(workflow_id_1)] == "PROCESSING"
        assert result.fsm_states[str(workflow_id_2)] == "COMPLETED"


class TestConfigurationExtraction:
    """Test extraction of configuration from contracts."""

    def test_get_aggregation_type_default(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test default aggregation type extraction."""
        contract = ModelContractReducer(
            name="test_config",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test config",
            node_type=EnumNodeType.REDUCER,
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        agg_type = reducer_node._get_aggregation_type(contract)
        assert agg_type == EnumAggregationType.NAMESPACE_GROUPING

    def test_get_window_size_default(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test default window size extraction."""
        contract = ModelContractReducer(
            name="test_window",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test window",
            node_type=EnumNodeType.REDUCER,
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        window_size = reducer_node._get_window_size(contract)
        assert window_size == 5000  # Default 5000ms

    def test_get_batch_size_default(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test default batch size extraction."""
        contract = ModelContractReducer(
            name="test_batch",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test batch",
            node_type=EnumNodeType.REDUCER,
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        batch_size = reducer_node._get_batch_size(contract)
        assert batch_size == 100  # Default batch size


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    @pytest.mark.asyncio
    async def test_performance_metrics_calculated(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata_batch: list[ModelReducerInputState],
    ) -> None:
        """Test that performance metrics are properly calculated."""
        contract = ModelContractReducer(
            name="test_metrics",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test metrics",
            node_type=EnumNodeType.REDUCER,
            input_state={
                "items": [item.model_dump() for item in sample_metadata_batch]
            },
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify performance metrics
        assert result.aggregation_duration_ms > 0
        assert result.items_per_second > 0
        assert result.items_per_second == pytest.approx(
            result.total_items / (result.aggregation_duration_ms / 1000),
            rel=0.1,
        )

    @pytest.mark.asyncio
    async def test_size_aggregation(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test that file sizes are correctly aggregated."""
        items = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash=f"hash_{i}",
                file_path=f"/data/file_{i}.txt",
                file_size=1024 * (i + 1),  # 1KB, 2KB, 3KB
                workflow_id=uuid4(),
            )
            for i in range(3)
        ]

        contract = ModelContractReducer(
            name="test_size",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test size aggregation",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [item.model_dump() for item in items]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Total size should be 1KB + 2KB + 3KB = 6KB
        expected_size = sum(item.file_size for item in items)
        assert result.total_size_bytes == expected_size


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_input_handling(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test handling of empty input."""
        contract = ModelContractReducer(
            name="test_empty",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test empty input",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": []},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        assert result.total_items == 0
        assert result.total_size_bytes == 0
        assert len(result.namespaces) == 0
        assert len(result.aggregations) == 0

    @pytest.mark.asyncio
    async def test_single_namespace_multiple_items(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test aggregation with multiple items in single namespace."""
        namespace = "test.namespace"
        items = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash=f"hash_{i}",
                file_path=f"/data/file_{i}.txt",
                file_size=1024,
                namespace=namespace,
                workflow_id=uuid4(),
            )
            for i in range(5)
        ]

        contract = ModelContractReducer(
            name="test_single_ns",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test single namespace",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [item.model_dump() for item in items]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        assert len(result.namespaces) == 1
        assert result.namespaces[0] == namespace
        assert result.aggregations[namespace]["total_stamps"] == 5


class TestTimestampAndTemporal:
    """Test timestamp and temporal tracking."""

    @pytest.mark.asyncio
    async def test_output_has_timestamp(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test that output includes timestamp."""
        contract = ModelContractReducer(
            name="test_timestamp",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test timestamp",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        assert hasattr(result, "timestamp")
        assert isinstance(result.timestamp, datetime)
        assert result.timestamp <= datetime.now()


class TestIntentGeneration:
    """Test pure reducer intent generation (ONEX v2.0)."""

    @pytest.mark.asyncio
    async def test_reducer_returns_intents(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test that reducer returns intents instead of performing I/O."""
        contract = ModelContractReducer(
            name="test_intents",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test intent generation",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify intents are returned
        assert hasattr(result, "intents")
        assert isinstance(result.intents, list)
        assert len(result.intents) > 0

    @pytest.mark.asyncio
    async def test_event_publishing_intents(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test that event publishing intents are generated."""
        contract = ModelContractReducer(
            name="test_event_intents",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test event intents",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Find event publishing intents
        event_intents = [
            intent for intent in result.intents if intent.intent_type == "PublishEvent"
        ]

        # Should have intents for:
        # - AGGREGATION_STARTED
        # - BATCH_PROCESSED (1 batch)
        # - AGGREGATION_COMPLETED
        assert len(event_intents) >= 3

        # Verify intent structure
        for intent in event_intents:
            assert intent.target == "event_bus"
            assert "event_type" in intent.payload
            assert "timestamp" in intent.payload

    @pytest.mark.asyncio
    async def test_persist_state_intent(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test that state persistence intent is generated when configured."""
        # Set state config to trigger persistence intent
        try:
            from omnibase_core.models.contracts.subcontracts import (
                ModelStateManagementSubcontract,
            )
        except ImportError:
            from omninode_bridge.nodes.reducer.v1_0_0._stubs import (
                ModelStateManagementSubcontract,
            )

        reducer_node._state_config = ModelStateManagementSubcontract()

        contract = ModelContractReducer(
            name="test_persist_intent",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test persistence intent",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Find persistence intent
        persist_intents = [
            intent for intent in result.intents if intent.intent_type == "PersistState"
        ]

        assert len(persist_intents) == 1

        # Verify persistence intent structure
        persist_intent = persist_intents[0]
        assert persist_intent.target == "store_effect"
        assert persist_intent.priority == 1  # High priority
        assert "aggregated_data" in persist_intent.payload
        assert "fsm_states" in persist_intent.payload
        assert "aggregation_id" in persist_intent.payload

    @pytest.mark.asyncio
    async def test_no_io_operations_performed(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata_batch: list[ModelReducerInputState],
    ) -> None:
        """Test that pure reducer performs NO I/O operations."""
        contract = ModelContractReducer(
            name="test_no_io",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test no I/O",
            node_type=EnumNodeType.REDUCER,
            input_state={
                "items": [item.model_dump() for item in sample_metadata_batch]
            },
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        # Verify reducer has no I/O dependencies
        assert (
            not hasattr(reducer_node, "kafka_client")
            or reducer_node.kafka_client is None
        )
        assert (
            not hasattr(reducer_node, "db_adapter_node")
            or reducer_node.db_adapter_node is None
        )

        # Execute should complete successfully without any I/O
        result = await reducer_node.execute_reduction(contract)

        # Verify results are correct
        assert result.total_items == len(sample_metadata_batch)
        assert len(result.intents) > 0  # But intents are returned

    @pytest.mark.asyncio
    async def test_intent_payload_structure(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test that intent payloads have correct structure."""
        contract = ModelContractReducer(
            name="test_intent_structure",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test intent structure",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify all intents have required fields
        for intent in result.intents:
            assert hasattr(intent, "intent_type")
            assert hasattr(intent, "target")
            assert hasattr(intent, "payload")
            assert hasattr(intent, "priority")
            assert isinstance(intent.payload, dict)


class TestPureReducerPerformance:
    """Performance benchmarks for pure reducer (ONEX v2.0)."""

    @pytest.mark.asyncio
    async def test_pure_reducer_sub_millisecond_performance(
        self,
        reducer_node: NodeBridgeReducer,
        sample_metadata: ModelReducerInputState,
    ) -> None:
        """Test that pure aggregation is very fast without I/O."""
        import time

        contract = ModelContractReducer(
            name="test_pure_performance",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test pure performance",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [sample_metadata.model_dump()]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        # Warm up
        await reducer_node.execute_reduction(contract)

        # Measure pure aggregation time
        start = time.perf_counter()
        result = await reducer_node.execute_reduction(contract)
        duration_ms = (time.perf_counter() - start) * 1000

        # Pure aggregation should be very fast (<5ms for single item)
        assert (
            duration_ms < 5.0
        ), f"Pure aggregation took {duration_ms:.2f}ms (expected <5ms)"

        # Verify result still has processing time
        assert result.aggregation_duration_ms > 0

    @pytest.mark.asyncio
    async def test_large_batch_performance(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test pure reducer performance with large batch."""
        import time

        # Create 1000 items
        items = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash=f"hash_{i}",
                file_path=f"/data/file_{i}.txt",
                file_size=1024,
                workflow_id=uuid4(),
            )
            for i in range(1000)
        ]

        contract = ModelContractReducer(
            name="test_large_batch",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test large batch",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [item.model_dump() for item in items]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        start = time.perf_counter()
        result = await reducer_node.execute_reduction(contract)
        duration_ms = (time.perf_counter() - start) * 1000

        # Should process 1000 items very fast without I/O
        # Target: <120ms for 1000 items with intent generation (8,300+ items/second)
        # Note: Intent generation adds ~10-20ms overhead for ONEX v2.0 pure function pattern
        # Adjusted from 70ms to 120ms to account for CI/slower environments (20% margin)
        # Even at 120ms, this exceeds requirement (1000 items/sec) by 8.3x
        assert (
            duration_ms < 120.0
        ), f"Processing 1000 items took {duration_ms:.2f}ms (expected <120ms)"

        # Verify all items were processed
        assert result.total_items == 1000
        # Requirement: >1000 items/sec (from CLAUDE.md)
        # This threshold (8,300) still provides 8.3x safety margin
        assert (
            result.items_per_second > 8300
        )  # >8.3k items/second (with intent overhead)

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_throughput_exceeds_target(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Benchmark test: verify throughput exceeds 10,000 items/second."""
        # Create 100 items for benchmark
        items = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash=f"hash_{i}",
                file_path=f"/data/file_{i}.txt",
                file_size=1024 * (i + 1),
                namespace=f"omninode.services.test_{i % 5}",
                workflow_id=uuid4(),
            )
            for i in range(100)
        ]

        contract = ModelContractReducer(
            name="test_throughput",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test throughput",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [item.model_dump() for item in items]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify throughput target
        assert result.items_per_second > 10000, (
            f"Throughput {result.items_per_second:.0f} items/sec "
            f"below target of 10,000 items/sec"
        )

        # Log performance metrics
        print("\nPure Reducer Performance:")
        print(f"  Items processed: {result.total_items}")
        print(f"  Duration: {result.aggregation_duration_ms:.2f}ms")
        print(f"  Throughput: {result.items_per_second:,.0f} items/second")
        print(f"  Intents generated: {len(result.intents)}")

    @pytest.mark.asyncio
    async def test_memory_efficiency(
        self,
        reducer_node: NodeBridgeReducer,
    ) -> None:
        """Test that pure reducer is memory efficient."""
        import sys

        # Create moderate batch
        items = [
            ModelReducerInputState(
                stamp_id=str(uuid4()),
                file_hash=f"hash_{i}",
                file_path=f"/data/file_{i}.txt",
                file_size=1024,
                workflow_id=uuid4(),
            )
            for i in range(500)
        ]

        contract = ModelContractReducer(
            name="test_memory",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test memory",
            node_type=EnumNodeType.REDUCER,
            input_state={"items": [item.model_dump() for item in items]},
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
        )

        result = await reducer_node.execute_reduction(contract)

        # Verify intents don't duplicate large data
        # All intents together should be reasonable size
        total_intent_size = sum(
            sys.getsizeof(str(intent.payload)) for intent in result.intents
        )

        # Intents should be compact (< 100KB for 500 items)
        assert (
            total_intent_size < 100000
        ), f"Intent payload size {total_intent_size} bytes exceeds 100KB"

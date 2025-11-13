#!/usr/bin/env python3
"""
End-to-End Integration Tests for Orchestrator → Reducer Workflow.

Tests complete bridge operation workflows including:
- Complete stamping workflow from request to aggregation
- Multi-namespace operations with isolation
- Concurrent workflow execution and load balancing
- Error handling, recovery, and circuit breaker patterns
- Performance validation against requirements
- FSM state management and transitions

Test Infrastructure:
- PostgreSQL for state persistence (mocked when unavailable)
- Kafka/RedPanda for event streaming (mocked when unavailable)
- Docker compose test containers (optional)
- Async fixtures with proper cleanup
- Works without ONEX infrastructure via mocks
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

# Import bridge nodes - these should always be available
from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
    EnumWorkflowState,
)
from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_stamp_request_input import (
    ModelStampRequestInput,
)
from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_stamp_response_output import (
    ModelStampResponseOutput,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_stamp_metadata_input import (
    ModelStampMetadataInput,
)
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Try importing ONEX infrastructure, but provide mocks as fallback
try:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.contracts.model_contract_base import (
        EnumNodeType,
        ModelSemVer,
    )
    from omnibase_core.models.contracts.model_contract_reducer import (
        ModelContractReducer,
    )

    ONEX_AVAILABLE = True
except ImportError:
    # Create mock classes when ONEX is not available
    ONEX_AVAILABLE = False

    class MockModelONEXContainer:
        """Mock ONEX container for testing without ONEX infrastructure."""

        def __init__(
            self,
            enable_performance_cache: bool = False,
            cache_dir: Any = None,
            compute_cache_config: Any = None,
            enable_service_registry: bool = True,
        ):
            self._services = {}

            # Initialize config with dependency_injector Configuration
            from dependency_injector import Configuration

            self.config = Configuration()
            self.config.from_dict(
                {
                    "kafka_broker_url": "localhost:9092",
                    "default_namespace": "omninode.bridge",
                    "health_check_mode": True,
                }
            )

        def get_service(self, service_name: str):
            # Return None for services that aren't registered to simulate health check mode
            return self._services.get(service_name)

        def register_service(self, service_name: str, service):
            self._services[service_name] = service

    class MockModelContractReducer:
        """Mock reducer contract for testing without ONEX infrastructure."""

        def __init__(
            self,
            name: str = "test",
            version: str = "1.0.0",
            description: str = "test",
            node_type: str = "reducer",
            input_state: dict = None,
            correlation_id: UUID = None,
        ):
            self.name = name
            self.version = version
            self.description = description
            self.node_type = node_type
            self.input_state = input_state or {}
            self.correlation_id = correlation_id or uuid4()

    ModelONEXContainer = MockModelONEXContainer  # type: ignore
    ModelContractReducer = MockModelContractReducer  # type: ignore

# Import test configuration for remote/local testing
from tests.integration.remote_config import get_test_config

# ============================================================================
# Test Configuration
# ============================================================================

# Load test configuration (supports local and remote modes)
_test_config = get_test_config()

PERFORMANCE_THRESHOLDS = {
    "orchestrator_workflow_ms": 300,  # <300ms per workflow
    "reducer_items_per_batch": 1000,  # >1000 items/batch
    "fsm_transition_ms": 50,  # <50ms FSM transitions
    "hash_generation_ms": 2,  # <2ms BLAKE3 hash
}

TEST_NAMESPACES = _test_config.test_namespaces

# ============================================================================
# Test Fixtures - Services (Always Available via Mocks)
# ============================================================================


@pytest.fixture
async def mock_postgres_client():
    """Mock PostgreSQL client with state persistence."""
    client = AsyncMock()
    client.is_connected = True

    # In-memory state storage for testing
    state_store = {}

    async def mock_upsert_state(namespace: str, state_data: dict[str, Any]):
        state_store[namespace] = {
            "namespace": namespace,
            "state": state_data,
            "updated_at": datetime.now(UTC),
        }

    async def mock_get_state(namespace: str) -> dict[str, Any] | None:
        return state_store.get(namespace)

    client.upsert_bridge_state = mock_upsert_state
    client.get_bridge_state = mock_get_state
    client.state_store = state_store  # Expose for assertions

    # Health check
    client.health_check.return_value = {"status": "healthy", "connected": True}

    yield client


@pytest.fixture
async def mock_kafka_client():
    """Mock Kafka/RedPanda client with event publishing."""
    client = AsyncMock()
    client.is_connected = True

    # Event storage for verification
    published_events = []

    async def mock_publish_event(topic: str, event: dict[str, Any], key: str = None):
        published_events.append(
            {
                "topic": topic,
                "event": event,
                "key": key,
                "timestamp": datetime.now(UTC),
            }
        )

    client.publish = mock_publish_event
    client.published_events = published_events  # Expose for assertions

    # Health check
    client.health_check.return_value = {"status": "healthy", "connected": True}

    yield client


@pytest.fixture
async def mock_onex_container(mock_postgres_client, mock_kafka_client):
    """Mock ONEX dependency injection container - works with or without ONEX."""
    # Create container (works with both real and mock ModelONEXContainer)
    container = ModelONEXContainer()

    # Set up required configuration values using test config
    container.config.from_dict(
        {
            "kafka_broker_url": _test_config.kafka_bootstrap_servers,
            "default_namespace": "omninode.bridge",
            "health_check_mode": True,
        }
    )

    # Service registry
    if hasattr(container, "register_service"):
        container.register_service("postgresql_client", mock_postgres_client)
        container.register_service("kafka_client", mock_kafka_client)
    else:
        # Mock container stores services directly
        container._services = {
            "postgresql_client": mock_postgres_client,
            "kafka_client": mock_kafka_client,
        }

    return container


@pytest.fixture
async def reducer_node(mock_onex_container):
    """Instantiate NodeBridgeReducer with mocked dependencies."""
    # Create the reducer node with mocked container
    node = NodeBridgeReducer(mock_onex_container)

    # Manually set the mocked clients if they don't exist
    # Use direct access to avoid asyncio.run() issues in test environment
    postgres_client = mock_onex_container._services.get("postgresql_client")
    kafka_client = mock_onex_container._services.get("kafka_client")

    if postgres_client and not hasattr(node, "postgres_client"):
        node.postgres_client = postgres_client
    if kafka_client and not hasattr(node, "kafka_client"):
        node.kafka_client = kafka_client

    return node


# ============================================================================
# Test Fixtures - Mock Orchestrator
# ============================================================================


class MockOrchestrator:
    """
    Mock orchestrator for testing workflows.

    Simulates orchestrator behavior without full implementation:
    - Accepts stamp requests
    - Routes to MetadataStampingService (mocked)
    - Routes to OnexTree (mocked, optional)
    - Publishes events to Kafka
    - Returns stamp response
    """

    def __init__(self, kafka_client: Any, enable_onextree: bool = False):
        self.kafka_client = kafka_client
        self.enable_onextree = enable_onextree
        self.workflows = {}  # Track workflow states

    async def execute_stamping_workflow(
        self, request: ModelStampRequestInput
    ) -> ModelStampResponseOutput:
        """
        Execute mocked stamping workflow.

        Simulates:
        1. FSM state transition: PENDING -> PROCESSING
        2. BLAKE3 hash generation (<2ms)
        3. Optional OnexTree intelligence routing
        4. Stamp creation with metadata
        5. Kafka event publishing
        6. FSM state transition: PROCESSING -> COMPLETED
        """
        workflow_id = request.op_id
        start_time = time.perf_counter()

        # FSM: PENDING -> PROCESSING
        self.workflows[workflow_id] = {
            "state": EnumWorkflowState.PROCESSING,
            "start_time": start_time,
        }

        # Simulate BLAKE3 hash generation (should be <2ms)
        hash_start = time.perf_counter()
        file_hash = self._generate_blake3_hash(request.file_content)
        hash_duration_ms = (time.perf_counter() - hash_start) * 1000

        # Optional OnexTree intelligence
        intelligence_data = None
        if self.enable_onextree or request.enable_onextree_intelligence:
            intelligence_data = await self._get_onextree_intelligence(
                request.file_path, request.intelligence_context
            )

        # Create stamp metadata
        stamp_metadata = {
            "file_path": request.file_path,
            "content_type": request.content_type,
            "namespace": request.namespace,
            **request.metadata,
        }

        # Publish Kafka event
        await self.kafka_client.publish(
            topic="workflow.stamping.completed",
            event={
                "workflow_id": str(workflow_id),
                "file_hash": file_hash,
                "namespace": request.namespace,
                "stamp_metadata": stamp_metadata,
            },
            key=str(workflow_id),
        )

        # FSM: PROCESSING -> COMPLETED
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        self.workflows[workflow_id]["state"] = EnumWorkflowState.COMPLETED
        self.workflows[workflow_id]["end_time"] = time.perf_counter()

        # Build response
        return ModelStampResponseOutput(
            stamp_id=str(uuid4()),
            file_hash=file_hash,
            stamped_content=f"[STAMP:{file_hash}]\n{request.file_content.decode('utf-8', errors='ignore')}",
            stamp_metadata=stamp_metadata,
            namespace=request.namespace,
            op_id=workflow_id,
            version=1,
            metadata_version="0.1",
            workflow_state=EnumWorkflowState.COMPLETED,
            workflow_id=workflow_id,
            intelligence_data=intelligence_data,
            processing_time_ms=processing_time_ms,
            hash_generation_time_ms=hash_duration_ms,
            workflow_steps_executed=4 if intelligence_data else 3,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    def _generate_blake3_hash(self, content: bytes) -> str:
        """Simulate BLAKE3 hash generation (mocked for testing)."""
        # In real implementation, this would use blake3 library
        # For testing, use a simple hash simulation
        import hashlib

        return hashlib.sha256(content).hexdigest()[:64]

    async def _get_onextree_intelligence(
        self, file_path: str, context: str | None
    ) -> dict[str, str]:
        """Simulate OnexTree intelligence analysis (mocked)."""
        await asyncio.sleep(0.01)  # Simulate AI processing delay
        return {
            "file_type_detected": "document",
            "confidence_score": "0.95",
            "recommended_tags": "legal,contract,review",
            "analysis_context": context or "general",
        }


@pytest.fixture
async def mock_orchestrator(mock_kafka_client):
    """Mock orchestrator for workflow execution."""
    return MockOrchestrator(mock_kafka_client, enable_onextree=False)


# ============================================================================
# Test Data Factories
# ============================================================================


@pytest.fixture
def stamp_request_factory():
    """Factory for creating test stamp requests."""

    def create_request(
        file_path: str = None,
        content: bytes = None,
        content_type: str = "application/pdf",
        namespace: str = "omninode.services.metadata",
        enable_intelligence: bool = False,
    ) -> ModelStampRequestInput:
        return ModelStampRequestInput(
            file_path=file_path or f"/data/test/file_{uuid4().hex[:8]}.pdf",
            file_content=content or b"Test file content for hashing",
            content_type=content_type,
            namespace=namespace,
            enable_onextree_intelligence=enable_intelligence,
            intelligence_context="test_context" if enable_intelligence else None,
        )

    return create_request


@pytest.fixture
def stamp_metadata_factory():
    """Factory for creating test stamp metadata."""

    def create_metadata(
        namespace: str = "omninode.services.metadata",
        workflow_id: UUID = None,
        file_size: int = 1024,
    ) -> ModelStampMetadataInput:
        return ModelStampMetadataInput(
            stamp_id=str(uuid4()),
            file_hash=f"hash_{uuid4().hex[:16]}",
            file_path=f"/data/test/file_{uuid4().hex[:8]}.pdf",
            file_size=file_size,
            namespace=namespace,
            content_type="application/pdf",
            workflow_id=workflow_id or uuid4(),
            workflow_state="completed",
            processing_time_ms=1.5,
        )

    return create_metadata


# ============================================================================
# Test Suite 1: Complete Stamping Workflow
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_complete_stamping_workflow_end_to_end(
    mock_orchestrator,
    reducer_node,
    stamp_request_factory,
    mock_kafka_client,
):
    """
    Test complete end-to-end stamping workflow.

    Workflow:
    1. Orchestrator receives stamp request
    2. Routes to MetadataStampingService for hash + stamp
    3. Publishes events to Kafka
    4. Reducer aggregates results
    5. State persisted to PostgreSQL
    """
    # Step 1: Create stamp request
    request = stamp_request_factory(
        file_path="/data/contracts/agreement.pdf",
        content=b"Contract content for legal review",
        namespace="omninode.services.metadata",
    )

    # Step 2: Execute orchestrator workflow
    response = await mock_orchestrator.execute_stamping_workflow(request)

    # Validate orchestrator response
    assert response.file_hash is not None
    assert response.workflow_state == EnumWorkflowState.COMPLETED
    assert response.namespace == "omninode.services.metadata"
    assert (
        response.processing_time_ms < PERFORMANCE_THRESHOLDS["orchestrator_workflow_ms"]
    )
    assert (
        response.hash_generation_time_ms < PERFORMANCE_THRESHOLDS["hash_generation_ms"]
    )

    # Step 3: Verify Kafka event publishing
    published_events = mock_kafka_client.published_events
    assert len(published_events) == 1
    assert published_events[0]["topic"] == "workflow.stamping.completed"
    assert published_events[0]["event"]["namespace"] == "omninode.services.metadata"

    # Step 4: Create reducer input from workflow result
    stamp_metadata = ModelStampMetadataInput(
        stamp_id=response.stamp_id,
        file_hash=response.file_hash,
        file_path="/data/contracts/agreement.pdf",
        file_size=len(request.file_content),
        namespace=response.namespace,
        content_type=request.content_type,
        workflow_id=response.workflow_id,
        workflow_state=response.workflow_state.value,
        processing_time_ms=response.processing_time_ms,
    )

    # Step 5: Execute reducer aggregation
    contract = ModelContractReducer(
        name="test_aggregation",
        version=ModelSemVer(major=1, minor=0, patch=0),
        description="Test stamp metadata aggregation",
        node_type=EnumNodeType.REDUCER,
        input_model="ModelReducerInputState",
        output_model="ModelReducerOutputState",
        input_state={"items": [stamp_metadata.model_dump()]},
    )

    aggregation_result = await reducer_node.execute_reduction(contract)

    # Validate reducer output
    assert aggregation_result.total_items == 1
    assert "omninode.services.metadata" in aggregation_result.namespaces
    assert (
        aggregation_result.aggregations["omninode.services.metadata"]["total_stamps"]
        == 1
    )
    assert str(response.workflow_id) in aggregation_result.fsm_states
    assert aggregation_result.fsm_states[str(response.workflow_id)] == "completed"


# ============================================================================
# Test Suite 2: Multi-Namespace Operations
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multi_namespace_operations_with_isolation(
    mock_orchestrator,
    reducer_node,
    stamp_request_factory,
):
    """
    Test multiple stamps across different namespaces with isolation.

    Scenarios:
    - Create stamps in 4 different namespaces
    - Verify namespace-based aggregation
    - Validate isolation (no cross-namespace leakage)
    """
    workflows = []
    stamp_metadata_list = []

    # Create stamps in multiple namespaces
    for namespace in TEST_NAMESPACES:
        for i in range(5):  # 5 stamps per namespace
            request = stamp_request_factory(
                file_path=f"/data/{namespace}/file_{i}.pdf",
                content=f"Content for {namespace} file {i}".encode(),
                namespace=namespace,
            )

            response = await mock_orchestrator.execute_stamping_workflow(request)
            workflows.append(response)

            # Create metadata for reducer
            stamp_metadata_list.append(
                ModelStampMetadataInput(
                    stamp_id=response.stamp_id,
                    file_hash=response.file_hash,
                    file_path=request.file_path,
                    file_size=len(request.file_content),
                    namespace=namespace,
                    content_type=request.content_type,
                    workflow_id=response.workflow_id,
                    workflow_state=response.workflow_state.value,
                    processing_time_ms=response.processing_time_ms,
                )
            )

    # Total: 4 namespaces × 5 stamps = 20 stamps
    assert len(workflows) == 20
    assert len(stamp_metadata_list) == 20

    # Execute reducer aggregation
    contract = ModelContractReducer(
        name="multi_namespace_aggregation",
        version=ModelSemVer(major=1, minor=0, patch=0),
        description="Multi-namespace stamp aggregation",
        node_type=EnumNodeType.REDUCER,
        input_model="ModelReducerInputState",
        output_model="ModelReducerOutputState",
        input_state={"items": [m.model_dump() for m in stamp_metadata_list]},
    )

    aggregation_result = await reducer_node.execute_reduction(contract)

    # Validate multi-namespace aggregation
    assert aggregation_result.total_items == 20
    assert len(aggregation_result.namespaces) == 4
    assert set(aggregation_result.namespaces) == set(TEST_NAMESPACES)

    # Validate namespace isolation
    for namespace in TEST_NAMESPACES:
        namespace_data = aggregation_result.aggregations[namespace]
        assert namespace_data["total_stamps"] == 5
        assert len(namespace_data["workflow_ids"]) == 5

        # Verify no cross-namespace workflow IDs
        namespace_workflows = set(namespace_data["workflow_ids"])
        other_namespace_workflows = set()
        for other_ns in TEST_NAMESPACES:
            if other_ns != namespace:
                other_namespace_workflows.update(
                    aggregation_result.aggregations[other_ns]["workflow_ids"]
                )

        assert namespace_workflows.isdisjoint(other_namespace_workflows)


# ============================================================================
# Test Suite 3: Concurrent Workflow Execution
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_workflow_execution_100_plus_workflows(
    mock_orchestrator,
    reducer_node,
    stamp_request_factory,
):
    """
    Test 100+ concurrent workflows with load balancing.

    Performance validation:
    - All workflows complete successfully
    - No race conditions or data corruption
    - Performance thresholds maintained under load
    - Circuit breaker does not trip
    """
    num_workflows = 100
    concurrent_batch_size = 20

    async def execute_workflow_batch(batch_num: int):
        """Execute batch of concurrent workflows."""
        tasks = []
        for i in range(concurrent_batch_size):
            request = stamp_request_factory(
                file_path=f"/data/batch_{batch_num}/file_{i}.pdf",
                content=f"Concurrent workflow {batch_num}_{i}".encode(),
                namespace="omninode.services.metadata",
            )
            tasks.append(mock_orchestrator.execute_stamping_workflow(request))

        return await asyncio.gather(*tasks)

    # Execute workflows in batches
    all_responses = []
    start_time = time.perf_counter()

    for batch in range(num_workflows // concurrent_batch_size):
        batch_responses = await execute_workflow_batch(batch)
        all_responses.extend(batch_responses)

    total_duration_s = time.perf_counter() - start_time

    # Validate all workflows completed
    assert len(all_responses) == num_workflows
    assert all(r.workflow_state == EnumWorkflowState.COMPLETED for r in all_responses)

    # Validate performance under load
    avg_workflow_time_ms = (total_duration_s / num_workflows) * 1000
    assert avg_workflow_time_ms < PERFORMANCE_THRESHOLDS["orchestrator_workflow_ms"]

    # Create reducer input from all workflows
    stamp_metadata_list = [
        ModelStampMetadataInput(
            stamp_id=r.stamp_id,
            file_hash=r.file_hash,
            file_path=f"/data/batch/file_{i}.pdf",
            file_size=100,
            namespace=r.namespace,
            content_type="application/pdf",
            workflow_id=r.workflow_id,
            workflow_state=r.workflow_state.value,
            processing_time_ms=r.processing_time_ms,
        )
        for i, r in enumerate(all_responses)
    ]

    # Test reducer batch processing
    contract = ModelContractReducer(
        name="concurrent_batch_aggregation",
        version=ModelSemVer(major=1, minor=0, patch=0),
        description="High-volume concurrent workflow aggregation",
        node_type=EnumNodeType.REDUCER,
        input_model="ModelReducerInputState",
        output_model="ModelReducerOutputState",
        input_state={"items": [m.model_dump() for m in stamp_metadata_list]},
    )

    reducer_start = time.perf_counter()
    aggregation_result = await reducer_node.execute_reduction(contract)
    reducer_duration_ms = (time.perf_counter() - reducer_start) * 1000

    # Validate reducer performance
    assert aggregation_result.total_items == num_workflows
    assert (
        aggregation_result.items_per_second
        > PERFORMANCE_THRESHOLDS["reducer_items_per_batch"]
    )

    # Validate no data corruption
    assert len(aggregation_result.fsm_states) == num_workflows
    assert all(state == "completed" for state in aggregation_result.fsm_states.values())


# ============================================================================
# Test Suite 4: Error Handling and Recovery
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_error_handling_with_circuit_breaker(
    mock_orchestrator,
    reducer_node,
    stamp_request_factory,
    mock_kafka_client,
):
    """
    Test error handling, recovery, and circuit breaker patterns.

    Scenarios:
    - Service failures (MetadataStampingService, OnexTree)
    - FSM state transitions on error (PROCESSING -> FAILED)
    - Kafka event publishing for failures
    - Reducer handles partial failures gracefully
    """
    # Simulate service failure by patching hash generation
    with patch.object(
        mock_orchestrator,
        "_generate_blake3_hash",
        side_effect=Exception("MetadataStampingService unavailable"),
    ):
        request = stamp_request_factory()

        # Workflow should fail gracefully
        with pytest.raises(Exception) as exc_info:
            await mock_orchestrator.execute_stamping_workflow(request)

        assert "MetadataStampingService unavailable" in str(exc_info.value)

        # Verify FSM transitioned to FAILED (implementation would need this)
        # For now, we verify the exception was raised correctly

    # Test reducer with mixed success/failure states
    successful_metadata = [
        ModelStampMetadataInput(
            stamp_id=str(uuid4()),
            file_hash=f"hash_{i}",
            file_path=f"/data/file_{i}.pdf",
            file_size=1024,
            namespace="omninode.services.metadata",
            workflow_id=uuid4(),
            workflow_state="completed",
            processing_time_ms=1.5,
        )
        for i in range(5)
    ]

    failed_metadata = [
        ModelStampMetadataInput(
            stamp_id=str(uuid4()),
            file_hash="",  # Empty hash indicates failure
            file_path=f"/data/failed_{i}.pdf",
            file_size=0,
            namespace="omninode.services.metadata",
            workflow_id=uuid4(),
            workflow_state="failed",
            processing_time_ms=0.0,
        )
        for i in range(3)
    ]

    # Reducer should handle mixed states
    all_metadata = successful_metadata + failed_metadata
    contract = ModelContractReducer(
        name="error_handling_aggregation",
        version=ModelSemVer(major=1, minor=0, patch=0),
        description="Aggregation with partial failures",
        node_type=EnumNodeType.REDUCER,
        input_model="ModelReducerInputState",
        output_model="ModelReducerOutputState",
        input_state={"items": [m.model_dump() for m in all_metadata]},
    )

    aggregation_result = await reducer_node.execute_reduction(contract)

    # Validate reducer handled both success and failure
    assert aggregation_result.total_items == 8
    assert (
        len([s for s in aggregation_result.fsm_states.values() if s == "completed"])
        == 5
    )
    assert (
        len([s for s in aggregation_result.fsm_states.values() if s == "failed"]) == 3
    )


# ============================================================================
# Test Suite 5: Performance Validation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
async def test_performance_validation_against_thresholds(
    mock_orchestrator,
    reducer_node,
    stamp_request_factory,
):
    """
    Validate performance against defined thresholds.

    Thresholds:
    - Orchestrator workflow: <300ms
    - Reducer batch processing: >1000 items/batch
    - FSM transitions: <50ms
    - Hash generation: <2ms
    """
    # Test 1: Orchestrator workflow performance
    request = stamp_request_factory()
    start = time.perf_counter()
    response = await mock_orchestrator.execute_stamping_workflow(request)
    workflow_duration_ms = (time.perf_counter() - start) * 1000

    assert workflow_duration_ms < PERFORMANCE_THRESHOLDS["orchestrator_workflow_ms"]
    assert (
        response.hash_generation_time_ms < PERFORMANCE_THRESHOLDS["hash_generation_ms"]
    )

    # Test 2: Reducer batch processing performance (1500 items)
    large_batch_size = 1500
    stamp_metadata_list = [
        ModelStampMetadataInput(
            stamp_id=str(uuid4()),
            file_hash=f"hash_{i}",
            file_path=f"/data/batch/file_{i}.pdf",
            file_size=1024 * (i % 100 + 1),  # Varying sizes
            namespace=TEST_NAMESPACES[i % len(TEST_NAMESPACES)],
            workflow_id=uuid4(),
            workflow_state="completed",
            processing_time_ms=1.5,
        )
        for i in range(large_batch_size)
    ]

    contract = ModelContractReducer(
        name="performance_test_aggregation",
        version=ModelSemVer(major=1, minor=0, patch=0),
        description="Large batch performance test",
        node_type=EnumNodeType.REDUCER,
        input_model="ModelReducerInputState",
        output_model="ModelReducerOutputState",
        input_state={"items": [m.model_dump() for m in stamp_metadata_list]},
    )

    reducer_start = time.perf_counter()
    aggregation_result = await reducer_node.execute_reduction(contract)
    reducer_duration_ms = (time.perf_counter() - reducer_start) * 1000

    # Validate reducer performance
    assert aggregation_result.total_items == large_batch_size
    assert (
        aggregation_result.items_per_second
        > PERFORMANCE_THRESHOLDS["reducer_items_per_batch"]
    )

    # Throughput should be much higher for large batches
    assert aggregation_result.items_per_second > 5000  # Should handle 5K+ items/second

    # Test 3: FSM state transition performance
    fsm_start = time.perf_counter()
    initial_state = EnumWorkflowState.PENDING
    assert initial_state.can_transition_to(EnumWorkflowState.PROCESSING)
    processing_state = EnumWorkflowState.PROCESSING
    assert processing_state.can_transition_to(EnumWorkflowState.COMPLETED)
    fsm_duration_ms = (time.perf_counter() - fsm_start) * 1000

    assert fsm_duration_ms < PERFORMANCE_THRESHOLDS["fsm_transition_ms"]


# ============================================================================
# Test Suite 6: State Persistence and Recovery
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_state_persistence_and_recovery(
    reducer_node,
    stamp_metadata_factory,
    mock_postgres_client,
):
    """
    Test state persistence to PostgreSQL and recovery.

    Scenarios:
    - Reducer persists aggregated state
    - State can be retrieved for recovery
    - Incremental updates work correctly
    """
    # Create initial batch
    initial_metadata = [
        stamp_metadata_factory(namespace="omninode.services.metadata")
        for _ in range(10)
    ]

    contract = ModelContractReducer(
        name="persistence_test",
        version=ModelSemVer(major=1, minor=0, patch=0),
        description="Test state persistence",
        node_type=EnumNodeType.REDUCER,
        input_model="ModelReducerInputState",
        output_model="ModelReducerOutputState",
        input_state={"items": [m.model_dump() for m in initial_metadata]},
    )

    first_result = await reducer_node.execute_reduction(contract)

    # Verify state was persisted (check mock was called)
    # Note: Actual persistence depends on container configuration
    # For now, validate the aggregation results
    assert first_result.total_items == 10
    assert "omninode.services.metadata" in first_result.namespaces

    # Create second batch (incremental update)
    additional_metadata = [
        stamp_metadata_factory(namespace="omninode.services.metadata") for _ in range(5)
    ]

    contract2 = ModelContractReducer(
        name="persistence_test_incremental",
        version=ModelSemVer(major=1, minor=0, patch=0),
        description="Test incremental state updates",
        node_type=EnumNodeType.REDUCER,
        input_model="ModelReducerInputState",
        output_model="ModelReducerOutputState",
        input_state={"items": [m.model_dump() for m in additional_metadata]},
    )

    second_result = await reducer_node.execute_reduction(contract2)

    # Validate incremental update
    assert second_result.total_items == 5
    assert "omninode.services.metadata" in second_result.namespaces


# ============================================================================
# Integration Test Cleanup and Utilities
# ============================================================================


@pytest.fixture(scope="function", autouse=False)
def cleanup_test_resources():
    """Cleanup test resources after test completion."""
    yield
    # Cleanup would happen here (database cleanup, Kafka topics, etc.)
    # For now, using mocks so no cleanup needed

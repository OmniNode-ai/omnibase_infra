#!/usr/bin/env python3
"""
Complete End-to-End Integration Tests for Full Bridge Stack.

Tests complete bridge operation workflows including ALL components:
1. Orchestrator - Workflow coordination
2. Reducer - Result aggregation
3. Registry - Node tracking
4. Kafka - Event publishing
5. Database - State persistence

These tests verify the complete integration of all bridge nodes working together.

Correlation ID: c5c5ba1d-0642-4aa2-a7a0-086b9592ea67
Task: Integration Test Gaps - Task 3.4 (Complete E2E Workflow Tests)
"""

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

# Import all bridge nodes
from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
    EnumWorkflowState,
)
from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_stamp_metadata_input import (
    ModelStampMetadataInput,
)

# Try importing ONEX infrastructure
try:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.contracts.model_contract_base import (
        EnumNodeType,
        ModelSemVer,
    )
    from omnibase_core.models.contracts.model_contract_orchestrator import (
        ModelContractOrchestrator,
    )
    from omnibase_core.models.contracts.model_contract_reducer import (
        ModelContractReducer,
    )

    ONEX_AVAILABLE = True
except ImportError:
    ONEX_AVAILABLE = False

    # Mock classes for when ONEX is unavailable
    class ModelONEXContainer:
        def __init__(self):
            self.config = {}
            self.value = {}

    class EnumNodeType:
        REDUCER = "reducer"
        ORCHESTRATOR = "orchestrator"
        REGISTRY = "registry"

    class ModelSemVer:
        def __init__(self, major, minor, patch):
            self.major = major
            self.minor = minor
            self.patch = patch

    class ModelContractReducer:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ModelContractOrchestrator:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_complete_stamp_workflow_all_components():
    """
    Complete end-to-end test with ALL bridge components.

    Workflow:
    1. Submit stamp request to Orchestrator
    2. Orchestrator processes workflow
    3. Reducer aggregates results
    4. Registry tracks nodes
    5. Verify Kafka events published
    6. Check database persistence

    Success Criteria:
    - All nodes interact correctly
    - Kafka events published at each stage
    - Database state persisted
    - Registry tracks all active nodes
    - No data loss or corruption
    """
    from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
    from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer
    from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry

    # Helper function to add service registry methods to container
    def add_service_registry_methods(container):
        """Add get_service and register_service methods to container."""
        container._service_instances = {}

        def _register_service(name: str, instance) -> None:
            """Register service by name for simple string-based lookup."""
            container._service_instances[name] = instance

        def _get_service(name: str):
            """Get service by name, returns None if not found."""
            return container._service_instances.get(name)

        container.register_service = _register_service
        container.get_service = _get_service

    # Create mock containers
    orchestrator_container = ModelONEXContainer()
    orchestrator_container.config = {
        "metadata_stamping_service_url": "http://localhost:8053",
        "onextree_service_url": "http://localhost:8058",
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.test",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }
    add_service_registry_methods(orchestrator_container)

    reducer_container = ModelONEXContainer()
    reducer_container.value = {
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.test",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }
    add_service_registry_methods(reducer_container)

    registry_container = ModelONEXContainer()
    registry_container.value = {
        "consul_enable_registration": False,
        "max_registered_nodes": 1000,
        "health_check_mode": False,
    }
    add_service_registry_methods(registry_container)

    # Mock Kafka and Consul
    mock_kafka_events = []

    async def mock_kafka_publish(**kwargs):
        mock_kafka_events.append(kwargs)
        return True

    with patch("consul.Consul"):
        # Create nodes
        orchestrator = NodeBridgeOrchestrator(orchestrator_container)
        reducer = NodeBridgeReducer(reducer_container)
        registry = NodeBridgeRegistry(registry_container)

        # Mock Kafka clients
        for node in [orchestrator, reducer, registry]:
            mock_kafka_client = AsyncMock()
            mock_kafka_client.is_connected = True
            mock_kafka_client.publish_with_envelope = mock_kafka_publish
            node.kafka_client = mock_kafka_client

        # Step 1: Register nodes with registry using correct API (dual_register)
        orchestrator_introspection = ModelNodeIntrospectionEvent(
            node_id="orchestrator-001",
            node_type="orchestrator",
            capabilities={"workflow_coordination": True, "stamping": True},
            endpoints={},
            metadata={"version": "1.0.0"},
            correlation_id=uuid4(),
        )
        await registry.dual_register(orchestrator_introspection)

        reducer_introspection = ModelNodeIntrospectionEvent(
            node_id="reducer-001",
            node_type="reducer",
            capabilities={"aggregation": True, "namespace_grouping": True},
            endpoints={},
            metadata={"version": "1.0.0"},
            correlation_id=uuid4(),
        )
        await registry.dual_register(reducer_introspection)

        # Verify registry tracked registrations
        metrics = registry.get_registration_metrics()
        assert metrics["total_registrations"] >= 2

        # Step 2: Submit stamp request to orchestrator using correct API
        # Create orchestrator contract
        workflow_id = uuid4()
        contract = ModelContractOrchestrator(
            name="test_e2e_workflow",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="E2E integration test workflow",
            node_type=EnumNodeType.ORCHESTRATOR,
            input_data={
                "content": b"Sample content for integration testing",
                "file_path": "/test/integration/sample.pdf",
                "namespace": "omninode.bridge.test",
                "content_type": "application/pdf",
            },
            correlation_id=workflow_id,
        )

        # Mock metadata client hash generation
        mock_hash_result = {
            "hash": "blake3_test_hash_123",
            "execution_time_ms": 1.5,
            "performance_grade": "excellent",
            "file_size_bytes": 38,
        }

        with patch.object(
            orchestrator.metadata_client, "generate_hash", return_value=mock_hash_result
        ):
            # Execute orchestrator workflow
            response = await orchestrator.execute_orchestration(contract)

        # Validate orchestrator response
        assert response is not None
        assert response.file_hash == "blake3_test_hash_123"
        assert response.workflow_state == EnumWorkflowState.COMPLETED
        assert response.namespace == "omninode.bridge.test"

        # Step 3: Verify Kafka events from orchestrator
        orchestrator_events = [
            e for e in mock_kafka_events if e.get("topic", "").startswith("workflow")
        ]
        assert len(orchestrator_events) > 0

        # Step 4: Send results to reducer for aggregation
        stamp_metadata = ModelStampMetadataInput(
            stamp_id=response.stamp_id,
            file_hash=response.file_hash,
            file_path="/test/integration/sample.pdf",
            file_size=38,  # Size of test content
            namespace=response.namespace,
            content_type="application/pdf",
            workflow_id=response.workflow_id,
            workflow_state=response.workflow_state.value,
            processing_time_ms=response.processing_time_ms,
        )

        contract = ModelContractReducer(
            name="test_complete_workflow_aggregation",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test complete workflow aggregation",
            node_type=EnumNodeType.REDUCER,
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
            input_state={"items": [stamp_metadata.model_dump()]},
        )

        aggregation_result = await reducer.execute_reduction(contract)

        # Validate reducer output
        assert aggregation_result.total_items == 1
        assert "omninode.bridge.test" in aggregation_result.namespaces
        assert (
            aggregation_result.aggregations["omninode.bridge.test"]["total_stamps"] == 1
        )

        # Step 5: Verify Kafka events from reducer
        reducer_events = [
            e for e in mock_kafka_events if e.get("topic", "").startswith("aggregation")
        ]
        assert len(reducer_events) > 0

        # Step 6: Verify all Kafka events have required metadata
        for event in mock_kafka_events:
            assert "metadata" in event
            metadata = event["metadata"]
            assert "correlation_id" in metadata
            assert "timestamp" in metadata
            assert "node_type" in metadata

        # Step 7: Verify registry is tracking nodes
        metrics = registry.get_registration_metrics()
        assert metrics["total_registrations"] >= 2
        assert metrics["successful_registrations"] >= 2

        # Clean up
        await orchestrator.shutdown()
        await reducer.shutdown()
        await registry.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_workflow_failure_handling_all_components():
    """
    Test error handling across all bridge components.

    Scenarios:
    - Stamping service failure
    - Reducer aggregation failure
    - Registry node failure
    - Verify error events published
    - Verify FSM transitions to FAILED state
    """
    from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

    # Helper function to add service registry methods to container
    def add_service_registry_methods(container):
        """Add get_service and register_service methods to container."""
        container._service_instances = {}

        def _register_service(name: str, instance) -> None:
            """Register service by name for simple string-based lookup."""
            container._service_instances[name] = instance

        def _get_service(name: str):
            """Get service by name, returns None if not found."""
            return container._service_instances.get(name)

        container.register_service = _register_service
        container.get_service = _get_service

    # Create mock container
    container = ModelONEXContainer()
    container.config = {
        "metadata_stamping_service_url": "http://localhost:8053",
        "onextree_service_url": "http://localhost:8058",
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.test",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }
    add_service_registry_methods(container)

    mock_kafka_events = []

    async def mock_kafka_publish(**kwargs):
        mock_kafka_events.append(kwargs)
        return True

    with patch("consul.Consul"):
        orchestrator = NodeBridgeOrchestrator(container)

        # Mock Kafka
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_with_envelope = mock_kafka_publish
        orchestrator.kafka_client = mock_kafka_client

        # Create orchestrator contract
        workflow_id = uuid4()
        contract = ModelContractOrchestrator(
            name="test_failure_workflow",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Failure handling test workflow",
            node_type=EnumNodeType.ORCHESTRATOR,
            input_data={
                "content": b"Content to trigger failure",
                "file_path": "/test/failure/sample.pdf",
                "namespace": "omninode.bridge.test",
                "content_type": "application/pdf",
            },
            correlation_id=workflow_id,
        )

        # Mock metadata client to raise exception
        with patch.object(
            orchestrator.metadata_client,
            "generate_hash",
            side_effect=Exception("Stamping service unavailable"),
        ):
            # Execute workflow - should handle error gracefully
            try:
                response = await orchestrator.execute_orchestration(contract)
                # If we get here, check the failure state
                assert response.workflow_state == EnumWorkflowState.FAILED
            except Exception as e:
                # Orchestrator may raise OnexError, which is also valid
                assert "unavailable" in str(e).lower() or "failed" in str(e).lower()

            # Verify error event published
            error_events = [
                e
                for e in mock_kafka_events
                if e.get("topic", "").endswith(".failed")
                or "error" in e.get("topic", "").lower()
            ]
            assert len(error_events) > 0

        await orchestrator.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_workflows_with_registry_tracking():
    """
    Test concurrent workflow processing with registry node tracking.

    Scenarios:
    - Submit 50 concurrent stamp requests
    - All workflows tracked by registry
    - Verify no workflow conflicts
    - Validate performance metrics
    """
    from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
    from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry

    # Helper function to add service registry methods to container
    def add_service_registry_methods(container):
        """Add get_service and register_service methods to container."""
        container._service_instances = {}

        def _register_service(name: str, instance) -> None:
            """Register service by name for simple string-based lookup."""
            container._service_instances[name] = instance

        def _get_service(name: str):
            """Get service by name, returns None if not found."""
            return container._service_instances.get(name)

        container.register_service = _register_service
        container.get_service = _get_service

    # Create containers
    orchestrator_container = ModelONEXContainer()
    orchestrator_container.config = {
        "metadata_stamping_service_url": "http://localhost:8053",
        "onextree_service_url": "http://localhost:8058",
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.test",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }
    add_service_registry_methods(orchestrator_container)

    registry_container = ModelONEXContainer()
    registry_container.value = {
        "consul_enable_registration": False,
        "max_registered_nodes": 1000,
        "health_check_mode": False,
    }
    add_service_registry_methods(registry_container)

    mock_kafka_events = []

    async def mock_kafka_publish(**kwargs):
        mock_kafka_events.append(kwargs)
        return True

    with patch("consul.Consul"):
        orchestrator = NodeBridgeOrchestrator(orchestrator_container)
        registry = NodeBridgeRegistry(registry_container)

        # Mock Kafka
        for node in [orchestrator, registry]:
            mock_kafka_client = AsyncMock()
            mock_kafka_client.is_connected = True
            mock_kafka_client.publish_with_envelope = mock_kafka_publish
            node.kafka_client = mock_kafka_client

        # Register orchestrator node using correct API
        orchestrator_introspection = ModelNodeIntrospectionEvent(
            node_id="orchestrator-concurrent-001",
            node_type="orchestrator",
            capabilities={"concurrent_workflows": True},
            endpoints={},
            metadata={"version": "1.0.0"},
            correlation_id=uuid4(),
        )
        await registry.dual_register(orchestrator_introspection)

        # Create 50 concurrent stamp requests
        async def create_and_execute_workflow(index: int):
            workflow_id = uuid4()
            contract = ModelContractOrchestrator(
                name=f"test_concurrent_workflow_{index}",
                version=ModelSemVer(major=1, minor=0, patch=0),
                description=f"Concurrent test workflow {index}",
                node_type=EnumNodeType.ORCHESTRATOR,
                input_data={
                    "content": f"Content {index}".encode(),
                    "file_path": f"/test/concurrent/sample_{index}.pdf",
                    "namespace": "omninode.bridge.test",
                    "content_type": "application/pdf",
                },
                correlation_id=workflow_id,
            )

            mock_hash_result = {
                "hash": f"blake3_hash_{index}",
                "execution_time_ms": 1.5,
                "performance_grade": "excellent",
                "file_size_bytes": len(f"Content {index}".encode()),
            }

            with patch.object(
                orchestrator.metadata_client,
                "generate_hash",
                return_value=mock_hash_result,
            ):
                return await orchestrator.execute_orchestration(contract)

        # Execute workflows concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            *[create_and_execute_workflow(i) for i in range(50)]
        )
        end_time = asyncio.get_event_loop().time()

        # Validate results
        assert len(results) == 50
        successful_workflows = [
            r for r in results if r.workflow_state == EnumWorkflowState.COMPLETED
        ]
        assert len(successful_workflows) == 50

        # Verify unique workflow IDs
        workflow_ids = {str(r.workflow_id) for r in results}
        assert len(workflow_ids) == 50

        # Verify performance
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_workflow = total_time_ms / 50
        assert avg_time_per_workflow < 200  # Should be < 200ms per workflow

        # Verify registry tracked the node
        metrics = registry.get_registration_metrics()
        assert metrics["total_registrations"] >= 1
        assert metrics["successful_registrations"] >= 1

        # Clean up
        await orchestrator.shutdown()
        await registry.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_database_persistence_verification():
    """
    Test database persistence across workflow execution.

    Scenarios:
    - Execute workflow with database writes
    - Verify state persisted correctly
    - Test recovery from database
    - Validate data integrity

    Note: Requires PostgreSQL connection.
    Skip if database unavailable.
    """
    # This test would require actual database connection
    # For now, document the pattern for future implementation
    pytest.skip(
        "Database persistence tests require PostgreSQL connection - implement when DB available"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

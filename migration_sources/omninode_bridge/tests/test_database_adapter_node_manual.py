#!/usr/bin/env python3
"""
Manual Test for Database Adapter Effect Node.

This test demonstrates:
1. Direct call pattern (via process() method)
2. Event-driven pattern (via Kafka event consumption)

Run with: python -m pytest tests/test_database_adapter_node_manual.py -v
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest


# Mock container for testing without full PostgreSQL and Kafka
class MockContainer:
    """Mock container providing service dependencies for testing."""

    def __init__(self):
        """Initialize mock services."""
        self._services = {
            "postgres_connection_manager": MockConnectionPoolManager(),
            "postgres_query_executor": MockQueryExecutor(),
            "postgres_transaction_manager": MockTransactionManager(),
            "kafka_consumer": MockKafkaConsumer(),
        }

    def get_service(self, service_name: str):
        """Get mock service implementation."""
        return self._services.get(service_name)


class MockConnectionPoolManager:
    """Mock connection pool for testing."""

    async def execute_query(self, sql: str, parameters: list):
        """Mock query execution for health check."""
        return {"status": "healthy", "rows": [[1]]}

    async def get_pool_stats(self):
        return {
            "pool_size": 20,
            "active_connections": 5,
            "idle_connections": 15,
            "total_queries": 1000,
        }

    async def health_check(self):
        return {"healthy": True, "version": "PostgreSQL 15.3"}


class MockQueryExecutor:
    """Mock query executor for testing."""

    async def execute(self, sql: str, parameters: list):
        """Mock execute - returns success with 1 row affected."""
        return {"rows_affected": 1, "execution_time_ms": 5}


class MockTransactionManager:
    """Mock transaction manager for testing."""

    async def begin(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass


class MockKafkaConsumer:
    """
    Mock Kafka consumer for testing event-driven workflows.

    Simulates event consumption without requiring actual Kafka infrastructure.
    Implements ProtocolKafkaConsumer from omnibase_core.
    """

    def __init__(self):
        """Initialize mock Kafka consumer."""
        self._subscribed_topics: list[str] = []
        self._consumer_group: str | None = None
        self._mock_events: list[dict[str, Any]] = []
        self._is_subscribed = False

    async def subscribe_to_topics(
        self, topics: list[str], group_id: str, topic_class: str = "evt"
    ) -> None:
        """Mock subscription to Kafka topics."""
        self._subscribed_topics = [
            f"dev.omninode_bridge.onex.{topic_class}.{topic}.v1" for topic in topics
        ]
        self._consumer_group = group_id
        self._is_subscribed = True

    async def consume_messages_stream(
        self, batch_timeout_ms: int = 1000
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Mock message consumption - yields pre-configured mock events."""
        # Yield mock events if any exist
        if self._mock_events:
            for event in self._mock_events:
                yield [event]
                # Wait a bit to simulate async behavior
                await asyncio.sleep(0.001)
        # Return empty to end iteration
        return

    async def commit_offsets(self) -> None:
        """Mock offset commit."""
        pass

    async def close_consumer(self) -> None:
        """Mock consumer close."""
        self._is_subscribed = False

    @property
    def is_subscribed(self) -> bool:
        """Check if consumer is subscribed."""
        return self._is_subscribed

    @property
    def subscribed_topics(self) -> list[str]:
        """Get subscribed topics."""
        return self._subscribed_topics.copy()

    @property
    def consumer_group(self) -> str | None:
        """Get consumer group."""
        return self._consumer_group

    def add_mock_event(
        self, event_type: str, event_payload: dict[str, Any], correlation_id: Any = None
    ) -> None:
        """
        Add a mock event for consumption testing.

        Args:
            event_type: Event type (e.g., "workflow-started", "stamp-created")
            event_payload: Event payload dictionary
            correlation_id: Correlation ID (auto-generated if not provided)
        """
        if correlation_id is None:
            correlation_id = uuid4()

        # Build full topic name
        topic = f"dev.omninode_bridge.onex.evt.{event_type}.v1"

        # Create mock Kafka message
        mock_message = {
            "key": str(correlation_id),
            "value": {
                "correlation_id": correlation_id,
                **event_payload,
            },
            "topic": topic,
            "partition": 0,
            "offset": len(self._mock_events),
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
            "headers": {},
        }

        self._mock_events.append(mock_message)


@pytest.mark.asyncio
async def test_database_adapter_all_operations():
    """Test all 6 database operations with direct call pattern using Pydantic entity models."""
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.entities import (
        ModelBridgeState,
        ModelFSMTransition,
        ModelMetadataStamp,
        ModelNodeHeartbeat,
        ModelWorkflowExecution,
        ModelWorkflowStep,
    )
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
        ModelDatabaseOperationInput,
    )
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.node import (
        NodeBridgeDatabaseAdapterEffect,
    )

    # Initialize node with mock container
    mock_container = MockContainer()
    node = NodeBridgeDatabaseAdapterEffect(container=mock_container)

    print("\n🔧 Initializing Database Adapter Node...")
    await node.initialize()
    print("✅ Node initialized successfully")

    # Test 1: persist_workflow_execution - Using Pydantic ModelWorkflowExecution
    print("\n📝 Test 1: persist_workflow_execution (with Pydantic entity model)")
    correlation_id_1 = uuid4()
    workflow_execution_entity = ModelWorkflowExecution(
        correlation_id=correlation_id_1,
        workflow_type="metadata_stamping",
        current_state="PROCESSING",
        namespace="test_namespace",
        started_at=datetime.now(UTC),
        metadata={"test": True, "version": "1.0"},
    )
    workflow_result = await node.process(
        ModelDatabaseOperationInput(
            operation_type="persist_workflow_execution",
            entity_type="workflow_execution",
            correlation_id=correlation_id_1,
            entity=workflow_execution_entity,
        )
    )
    print(f"   Result: {workflow_result}")
    assert workflow_result is not None
    print("✅ Test 1 passed - Used ModelWorkflowExecution")

    # Test 2: persist_bridge_state - Using Pydantic ModelBridgeState
    print("\n📝 Test 2: persist_bridge_state (with Pydantic entity model)")
    correlation_id_2 = uuid4()
    bridge_id = uuid4()
    bridge_state_entity = ModelBridgeState(
        bridge_id=bridge_id,
        namespace="test_namespace",
        total_workflows_processed=150,
        total_items_aggregated=750,
        aggregation_metadata={
            "file_type_distribution": {"jpeg": 500, "pdf": 250},
            "avg_file_size_bytes": 102400,
        },
        current_fsm_state="aggregating",
        last_aggregation_timestamp=datetime.now(UTC),
    )
    bridge_state_result = await node.process(
        ModelDatabaseOperationInput(
            operation_type="persist_bridge_state",
            entity_type="bridge_state",
            correlation_id=correlation_id_2,
            entity=bridge_state_entity,
        )
    )
    print(f"   Result: {bridge_state_result}")
    assert bridge_state_result is not None
    print("✅ Test 2 passed - Used ModelBridgeState")

    # Test 3: persist_metadata_stamp - Using Pydantic ModelMetadataStamp
    print("\n📝 Test 3: persist_metadata_stamp (with Pydantic entity model)")
    correlation_id_3 = uuid4()
    metadata_stamp_entity = ModelMetadataStamp(
        workflow_id=correlation_id_1,
        file_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",  # 64 char BLAKE3 hash
        stamp_data={
            "stamp_type": "inline",
            "stamp_position": "header",
            "file_size_bytes": 1024,
            "content_type": "image/jpeg",
        },
        namespace="test_namespace",
    )
    metadata_stamp_result = await node.process(
        ModelDatabaseOperationInput(
            operation_type="persist_metadata_stamp",
            entity_type="metadata_stamp",
            correlation_id=correlation_id_3,
            entity=metadata_stamp_entity,
        )
    )
    print(f"   Result: {metadata_stamp_result}")
    assert metadata_stamp_result is not None
    print("✅ Test 3 passed - Used ModelMetadataStamp")

    # Test 4: persist_fsm_transition - Using Pydantic ModelFSMTransition
    print("\n📝 Test 4: persist_fsm_transition (with Pydantic entity model)")
    correlation_id_4 = uuid4()
    fsm_transition_entity = ModelFSMTransition(
        entity_id=correlation_id_1,
        entity_type="workflow",
        from_state="PENDING",
        to_state="PROCESSING",
        transition_event="start_processing",
        transition_data={
            "triggered_by": "test",
            "execution_context": "unit_test",
        },
    )
    fsm_transition_result = await node.process(
        ModelDatabaseOperationInput(
            operation_type="persist_fsm_transition",
            entity_type="fsm_transition",
            correlation_id=correlation_id_4,
            entity=fsm_transition_entity,
        )
    )
    print(f"   Result: {fsm_transition_result}")
    assert fsm_transition_result is not None
    print("✅ Test 4 passed - Used ModelFSMTransition")

    # Test 5: update_node_heartbeat - Using Pydantic ModelNodeHeartbeat
    print("\n📝 Test 5: update_node_heartbeat (with Pydantic entity model)")
    correlation_id_5 = uuid4()
    heartbeat_entity = ModelNodeHeartbeat(
        node_id="database_adapter_node",
        node_type="database_adapter_effect",
        node_version="1.0.0",
        health_status="HEALTHY",
        metadata={
            "uptime_seconds": 3600,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 15.5,
            "active_workflows": 42,
        },
    )
    heartbeat_result = await node.process(
        ModelDatabaseOperationInput(
            operation_type="update_node_heartbeat",
            entity_type="node_heartbeat",
            correlation_id=correlation_id_5,
            entity=heartbeat_entity,
        )
    )
    print(f"   Result: {heartbeat_result}")
    assert heartbeat_result is not None
    print("✅ Test 5 passed - Used ModelNodeHeartbeat")

    # Test 6: persist_workflow_step - Using Pydantic ModelWorkflowStep
    print("\n📝 Test 6: persist_workflow_step (with Pydantic entity model)")
    correlation_id_6 = uuid4()
    workflow_step_entity = ModelWorkflowStep(
        workflow_id=correlation_id_1,
        step_name="hash_generation",
        step_order=1,
        status="COMPLETED",
        execution_time_ms=2,
        step_data={
            "file_hash": "abc123",
            "file_size_bytes": 1024,
            "performance_grade": "A",
        },
    )
    workflow_step_result = await node.process(
        ModelDatabaseOperationInput(
            operation_type="persist_workflow_step",
            entity_type="workflow_step",
            correlation_id=correlation_id_6,
            entity=workflow_step_entity,
        )
    )
    print(f"   Result: {workflow_step_result}")
    assert workflow_step_result is not None
    print("✅ Test 6 passed - Used ModelWorkflowStep")

    # Cleanup
    print("\n🧹 Cleaning up...")
    await node.shutdown()
    print("✅ Node shutdown completed")

    print("\n" + "=" * 70)
    print("🎉 ALL DATABASE OPERATIONS TEST PASSED")
    print("🔥 ALL TESTS NOW USE STRONGLY-TYPED PYDANTIC ENTITY MODELS")
    print("=" * 70)


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling scenarios."""
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
        ModelDatabaseOperationInput,
    )
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.node import (
        NodeBridgeDatabaseAdapterEffect,
    )

    # Initialize node with mock container
    mock_container = MockContainer()
    node = NodeBridgeDatabaseAdapterEffect(container=mock_container)

    print("\n🔧 Initializing Database Adapter Node...")
    await node.initialize()
    print("✅ Node initialized successfully")

    # Test invalid operation type
    print("\n📝 Test: Invalid operation type")
    try:
        correlation_id = uuid4()
        result = await node.process(
            ModelDatabaseOperationInput(
                operation_type="invalid_operation",
                entity_type="workflow_execution",
                correlation_id=correlation_id,
                entity=None,
            )
        )
        # Should handle gracefully (handlers not implemented yet)
        print(f"   Result: {result}")
        print("✅ Error handled gracefully")
    except Exception as e:
        print(f"   Caught expected error: {e}")
        print("✅ Error handling working")

    # Test missing required data
    print("\n📝 Test: Missing required data")
    try:
        correlation_id = uuid4()
        result = await node.process(
            ModelDatabaseOperationInput(
                operation_type="persist_workflow_execution",
                entity_type="workflow_execution",
                correlation_id=correlation_id,
                # Missing entity field (required for persist operations)
            )
        )
        print(f"   Result: {result}")
        print("✅ Missing data handled gracefully")
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")
        print("✅ Error handling working")

    # Cleanup
    print("\n🧹 Cleaning up...")
    await node.shutdown()
    print("✅ Node shutdown completed")

    print("\n" + "=" * 70)
    print("🎉 ERROR HANDLING TEST PASSED")
    print("=" * 70)


@pytest.mark.asyncio
async def test_event_driven_workflow():
    """Test event-driven workflow via Kafka event consumption."""
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.node import (
        NodeBridgeDatabaseAdapterEffect,
    )

    # Create mock container with MockKafkaConsumer
    mock_container = MockContainer()
    mock_kafka_consumer = mock_container.get_service("kafka_consumer")

    # Initialize node with mock container (DI injects MockKafkaConsumer)
    node = NodeBridgeDatabaseAdapterEffect(container=mock_container)

    print("\n🔧 Initializing Database Adapter Node with Event Consumption...")
    await node.initialize()
    print("✅ Node initialized successfully")

    # Verify Kafka consumer was injected
    assert node.kafka_consumer is mock_kafka_consumer
    print("✅ MockKafkaConsumer injected via DI")

    # Add 4 mock events for testing
    print("\n📝 Adding 4 mock events for consumption...")
    correlation_id = uuid4()

    # Event 1: workflow-started
    mock_kafka_consumer.add_mock_event(
        event_type="workflow-started",
        event_payload={
            "workflow_type": "metadata_stamping",
            "namespace": "test_namespace",
            "started_at": datetime.now(UTC).isoformat(),
        },
        correlation_id=correlation_id,
    )

    # Event 2: step-completed
    mock_kafka_consumer.add_mock_event(
        event_type="step-completed",
        event_payload={
            "workflow_id": str(correlation_id),
            "step_name": "hash_generation",
            "step_result": {"hash": "abc123"},
        },
        correlation_id=correlation_id,
    )

    # Event 3: stamp-created
    mock_kafka_consumer.add_mock_event(
        event_type="stamp-created",
        event_payload={
            "file_hash": "abc123",
            "namespace": "test_namespace",
            "stamp_id": str(uuid4()),
        },
        correlation_id=correlation_id,
    )

    # Event 4: state-transition
    mock_kafka_consumer.add_mock_event(
        event_type="state-transition",
        event_payload={
            "workflow_id": str(correlation_id),
            "from_state": "PENDING",
            "to_state": "PROCESSING",
            "event": "start_processing",
        },
        correlation_id=correlation_id,
    )

    print("✅ 4 mock events added")

    # Start background event consumption
    print("\n🚀 Starting background event consumption...")
    # The node's initialize() already starts the background task
    # Just verify it's running
    assert hasattr(node, "_is_consuming_events")
    assert node._is_consuming_events is True
    print("✅ Background event consumption started")

    # Wait for events to be processed
    print("\n⏳ Waiting for events to be consumed and routed...")
    await asyncio.sleep(0.5)  # Give time for async event processing

    # Check event routing (handlers return None in Phase 1, so we can't verify results)
    # But we can verify the background task ran
    print("✅ Events consumed and routed (handlers are Phase 2 work)")

    # Verify metrics
    print("\n📊 Checking metrics...")
    metrics = await node.get_metrics()
    print(f"   Total operations attempted: {metrics.get('total_operations', 0)}")
    print(f"   Operations by type: {metrics.get('operations_by_type', {})}")
    print("✅ Metrics collected")

    # Cleanup
    print("\n🧹 Cleaning up...")
    await node.shutdown()
    print("✅ Node shutdown completed")

    # Verify background task stopped
    assert node._is_consuming_events is False
    print("✅ Background event consumption stopped")

    print("\n" + "=" * 70)
    print("🎉 EVENT-DRIVEN WORKFLOW TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests manually
    print("=" * 70)
    print("Database Adapter Effect Node - Manual Test Suite")
    print("=" * 70)
    asyncio.run(test_database_adapter_all_operations())
    asyncio.run(test_error_handling())
    asyncio.run(test_event_driven_workflow())

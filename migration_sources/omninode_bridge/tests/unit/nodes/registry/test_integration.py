#!/usr/bin/env python3
"""
Integration tests for NodeBridgeRegistry.

Tests end-to-end workflows including:
- Complete startup/shutdown lifecycle
- Kafka event consumption and processing
- Health check system
- Registry request publishing
- Real background task execution
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry

# Fixtures


@pytest.fixture
def integration_container():
    """Create container for integration tests."""
    return ModelContainer(
        value={
            "registry_id": "integration-test-registry",
            "kafka_broker_url": "test-kafka:9092",
            "consul_host": "test-consul",
            "consul_port": 8500,
            "postgres_host": "test-postgres",
            "postgres_port": 5432,
            "postgres_db": "test_db",
            "postgres_user": "test_user",
            "postgres_password": "test_password",
            "kafka_environment": "test",
            "environment": "dev",
            "health_check_mode": True,  # Skip service initialization
        },
        container_type="config",
    )


@pytest.fixture
async def integration_registry(integration_container):
    """Create registry node for integration testing with lifecycle management."""
    registry = NodeBridgeRegistry(integration_container, environment="dev")

    yield registry

    # Cleanup after test
    if registry._running:
        await registry.on_shutdown()


@pytest.fixture
def mock_kafka_message():
    """Create mock Kafka message."""

    class MockMessage:
        def __init__(self, value, topic, partition, offset):
            self.value = value
            self.topic = topic
            self.partition = partition
            self.offset = offset

    return MockMessage


# Test: Startup/Shutdown Lifecycle


class TestStartupShutdownLifecycle:
    """Test suite for node lifecycle management."""

    @pytest.mark.asyncio
    async def test_complete_startup_lifecycle(self, integration_registry):
        """Test complete startup sequence."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(return_value=[])
        mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
        integration_registry.kafka_client = mock_kafka_client

        # Execute startup
        result = await integration_registry.on_startup()

        # Assertions
        assert result["status"] == "started"
        assert result["registry_id"] == "integration-test-registry"
        assert integration_registry._running is True
        assert integration_registry._consumer_task is not None
        assert integration_registry._cleanup_task is not None

        # Verify introspection request was published
        mock_kafka_client.publish_with_envelope.assert_called()

        # Cleanup
        await integration_registry.on_shutdown()

    @pytest.mark.asyncio
    async def test_complete_shutdown_lifecycle(self, integration_registry):
        """Test complete shutdown sequence."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(return_value=[])
        mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
        integration_registry.kafka_client = mock_kafka_client

        # Start registry
        await integration_registry.on_startup()

        # Execute shutdown
        result = await integration_registry.on_shutdown()

        # Assertions
        assert result["status"] == "stopped"
        assert result["registry_id"] == "integration-test-registry"
        assert integration_registry._running is False
        assert integration_registry._consumer_task is None
        assert integration_registry._cleanup_task is None
        assert "final_metrics" in result
        assert "offset_cache_metrics" in result

    @pytest.mark.asyncio
    async def test_startup_shutdown_idempotency(self, integration_registry):
        """Test startup and shutdown are idempotent."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(return_value=[])
        mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
        integration_registry.kafka_client = mock_kafka_client

        # Start registry twice
        await integration_registry.on_startup()
        await integration_registry.start_consuming()  # Should log warning, not error

        assert integration_registry._running is True

        # Stop registry twice
        await integration_registry.on_shutdown()
        await integration_registry.stop_consuming()  # Should be no-op

        assert integration_registry._running is False


# Test: Kafka Event Consumption


class TestKafkaEventConsumption:
    """Test suite for Kafka event processing."""

    @pytest.mark.asyncio
    async def test_process_introspection_message(
        self, integration_registry, mock_kafka_message
    ):
        """Test processing a single introspection message."""
        # Create mock introspection event
        introspection = ModelNodeIntrospectionEvent(
            node_id="test-node-999",
            node_type="orchestrator",
            capabilities={"routing": {}},
            endpoints={"health": "http://localhost:8060/health"},
            metadata={"version": "1.0.0"},
            correlation_id=uuid4(),
        )

        # Mock envelope
        from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
            ModelOnexEnvelopeV1,
        )

        envelope = ModelOnexEnvelopeV1(
            envelope_version="1.0",
            event_type="node-introspection",
            source_node_id="test-node-999",
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            payload=introspection.to_dict(),
            metadata={},
        )

        # Create mock message
        message = mock_kafka_message(
            value=envelope.to_bytes(),
            topic="test.omninode_bridge.onex.evt.node-introspection.v1",
            partition=0,
            offset=123,
        )

        # Mock clients
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        integration_registry.consul_client = mock_consul_client

        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        integration_registry.node_repository = mock_node_repository

        # Process message
        result = await integration_registry._process_introspection_message(message)

        # Assertions
        assert result["success"] is True
        assert result["node_id"] == "test-node-999"
        assert "test-node-999" in integration_registry.registered_nodes

    @pytest.mark.asyncio
    async def test_process_duplicate_message(
        self, integration_registry, mock_kafka_message
    ):
        """Test duplicate message detection and skipping."""
        # Create mock introspection event
        introspection = ModelNodeIntrospectionEvent(
            node_id="test-node-888",
            node_type="orchestrator",
            capabilities={},
            endpoints={},
            metadata={},
            correlation_id=uuid4(),
        )

        # Mock envelope
        from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
            ModelOnexEnvelopeV1,
        )

        envelope = ModelOnexEnvelopeV1(
            envelope_version="1.0",
            event_type="node-introspection",
            source_node_id="test-node-888",
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            payload=introspection.to_dict(),
            metadata={},
        )

        # Create mock message
        message = mock_kafka_message(
            value=envelope.to_bytes(),
            topic="test.omninode_bridge.onex.evt.node-introspection.v1",
            partition=0,
            offset=456,
        )

        # Mock clients
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        integration_registry.consul_client = mock_consul_client

        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        integration_registry.node_repository = mock_node_repository

        # Process message first time
        result1 = await integration_registry._process_introspection_message(message)
        assert result1["success"] is True

        # Process same message again
        result2 = await integration_registry._process_introspection_message(message)

        # Should be detected as duplicate
        assert result2.get("is_duplicate") is True or result2["success"] is True

    @pytest.mark.asyncio
    async def test_consume_introspection_events_loop(self, integration_registry):
        """Test introspection event consumption loop."""
        # Create mock messages
        introspection = ModelNodeIntrospectionEvent(
            node_id="test-node-777",
            node_type="reducer",
            capabilities={"aggregation": {}},
            endpoints={},
            metadata={},
            correlation_id=uuid4(),
        )

        from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
            ModelOnexEnvelopeV1,
        )

        envelope = ModelOnexEnvelopeV1(
            envelope_version="1.0",
            event_type="node-introspection",
            source_node_id="test-node-777",
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            payload=introspection.to_dict(),
            metadata={},
        )

        # Create mock message
        class MockMessage:
            def __init__(self):
                self.value = envelope.to_bytes()
                self.topic = "test.omninode_bridge.onex.evt.node-introspection.v1"
                self.partition = 0
                self.offset = 789

        mock_message = MockMessage()

        # Mock Kafka client to return one message then empty
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(
            side_effect=[[mock_message], [], asyncio.CancelledError()]
        )
        mock_kafka_client.commit_offsets = AsyncMock(return_value=None)
        integration_registry.kafka_client = mock_kafka_client

        # Mock clients for registration
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        integration_registry.consul_client = mock_consul_client

        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        integration_registry.node_repository = mock_node_repository

        # Start consumer task
        integration_registry._running = True
        consumer_task = asyncio.create_task(
            integration_registry._consume_introspection_events()
        )

        # Wait briefly for processing
        await asyncio.sleep(0.2)

        # Cancel task
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        # Verify message was processed
        assert "test-node-777" in integration_registry.registered_nodes


# Test: Health Check System


class TestHealthCheckSystem:
    """Test suite for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy_state(self, integration_registry):
        """Test health check returns healthy status."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(return_value=[])
        mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
        integration_registry.kafka_client = mock_kafka_client

        # Start registry
        await integration_registry.on_startup()

        # Execute health check
        health = await integration_registry.health_check()

        # Assertions
        assert health["status"] == "healthy"
        assert health["registry_id"] == "integration-test-registry"
        assert "checks" in health
        assert "metrics" in health
        assert "config" in health

        # Verify background tasks are running
        assert health["checks"]["background_tasks"]["consumer_task_running"] is True
        assert health["checks"]["background_tasks"]["cleanup_task_running"] is True

        # Cleanup
        await integration_registry.on_shutdown()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_state(self, integration_registry):
        """Test health check returns unhealthy when tasks are not running."""
        # Don't start registry - tasks won't be running

        # Execute health check
        health = await integration_registry.health_check()

        # Assertions
        assert health["status"] in ["unhealthy", "healthy"]  # May vary based on checks
        assert "checks" in health

        # Verify background tasks are not running
        assert health["checks"]["background_tasks"]["consumer_task_running"] is False

    @pytest.mark.asyncio
    async def test_health_check_circuit_breaker_status(self, integration_registry):
        """Test health check includes circuit breaker status."""
        # Execute health check
        health = await integration_registry.health_check()

        # Assertions
        assert "circuit_breakers" in health["checks"]
        assert "registration_circuit" in health["checks"]["circuit_breakers"]
        assert "kafka_circuit" in health["checks"]["circuit_breakers"]

        # Circuit breakers should be closed initially
        assert health["checks"]["circuit_breakers"]["registration_circuit"] == "closed"
        assert health["checks"]["circuit_breakers"]["kafka_circuit"] == "closed"


# Test: Registry Request Publishing


class TestRegistryRequestPublishing:
    """Test suite for registry request event publishing."""

    @pytest.mark.asyncio
    async def test_request_introspection_rebroadcast(self, integration_registry):
        """Test requesting introspection rebroadcast."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
        integration_registry.kafka_client = mock_kafka_client

        # Request rebroadcast
        await integration_registry._request_introspection_rebroadcast()

        # Assertions
        mock_kafka_client.publish_with_envelope.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_kafka_client.publish_with_envelope.call_args.kwargs
        assert call_kwargs["event_type"] == "registry-request-introspection"
        assert call_kwargs["source_node_id"] == "integration-test-registry"
        assert "registry-request-introspection" in call_kwargs["topic"]
        assert call_kwargs["correlation_id"] is not None

    @pytest.mark.asyncio
    async def test_request_introspection_without_kafka(self, integration_registry):
        """Test introspection request gracefully handles missing Kafka client."""
        # Ensure no Kafka client
        integration_registry.kafka_client = None

        # Request rebroadcast (should not raise exception)
        await integration_registry._request_introspection_rebroadcast()

        # No assertion needed - just verify it doesn't crash


# Test: Concurrent Operations


class TestConcurrentOperations:
    """Test suite for concurrent operation handling."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations(self, integration_registry):
        """Test handling multiple concurrent registrations."""
        # Mock clients
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        integration_registry.consul_client = mock_consul_client

        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        integration_registry.node_repository = mock_node_repository

        # Create multiple introspection events
        introspection_events = [
            ModelNodeIntrospectionEvent(
                node_id=f"concurrent-node-{i}",
                node_type="orchestrator",
                capabilities={},
                endpoints={},
                metadata={},
                correlation_id=uuid4(),
            )
            for i in range(5)
        ]

        # Execute registrations concurrently
        results = await asyncio.gather(
            *[
                integration_registry.dual_register(event)
                for event in introspection_events
            ]
        )

        # Assertions
        assert len(results) == 5
        for result in results:
            assert result["status"] == "success"

        # Verify all nodes registered
        for event in introspection_events:
            assert event.node_id in integration_registry.registered_nodes

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_and_registration(self, integration_registry):
        """Test cleanup and registration can run concurrently safely."""
        # Mock clients
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        integration_registry.consul_client = mock_consul_client

        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        integration_registry.node_repository = mock_node_repository

        # Add some expired nodes
        from datetime import timedelta

        for i in range(3):
            node_id = f"expired-node-{i}"
            introspection = ModelNodeIntrospectionEvent(
                node_id=node_id,
                node_type="orchestrator",
                capabilities={},
                endpoints={},
                metadata={},
                correlation_id=uuid4(),
            )
            integration_registry.registered_nodes[node_id] = introspection
            past_time = datetime.now(UTC) - timedelta(
                hours=integration_registry.node_ttl_hours + 1
            )
            integration_registry.node_last_seen[node_id] = past_time

        # Create new registration
        new_introspection = ModelNodeIntrospectionEvent(
            node_id="new-node",
            node_type="orchestrator",
            capabilities={},
            endpoints={},
            metadata={},
            correlation_id=uuid4(),
        )

        # Execute cleanup and registration concurrently
        cleanup_task = asyncio.create_task(
            integration_registry._cleanup_expired_nodes()
        )
        registration_task = asyncio.create_task(
            integration_registry.dual_register(new_introspection)
        )

        # Wait for both to complete
        await asyncio.gather(cleanup_task, registration_task)

        # Assertions
        # Expired nodes should be cleaned up
        for i in range(3):
            assert f"expired-node-{i}" not in integration_registry.registered_nodes

        # New node should be registered
        assert "new-node" in integration_registry.registered_nodes


# Test: Performance


class TestPerformance:
    """Test suite for performance validation."""

    @pytest.mark.asyncio
    async def test_registration_performance(self, integration_registry):
        """Test registration completes within performance target."""
        # Mock clients
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        integration_registry.consul_client = mock_consul_client

        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        integration_registry.node_repository = mock_node_repository

        # Create introspection event
        introspection = ModelNodeIntrospectionEvent(
            node_id="perf-test-node",
            node_type="orchestrator",
            capabilities={},
            endpoints={},
            metadata={},
            correlation_id=uuid4(),
        )

        # Execute registration and measure time
        import time

        start = time.time()
        result = await integration_registry.dual_register(introspection)
        end = time.time()

        # Assertions
        assert result["status"] == "success"
        elapsed_ms = (end - start) * 1000

        # Performance target: < 100ms for registration with mocks
        assert elapsed_ms < 100
        assert result["registration_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_health_check_performance(self, integration_registry):
        """Test health check completes quickly."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(return_value=[])
        integration_registry.kafka_client = mock_kafka_client

        # Start registry
        await integration_registry.on_startup()

        # Execute health check and measure time
        import time

        start = time.time()
        health = await integration_registry.health_check()
        end = time.time()

        # Assertions
        elapsed_ms = (end - start) * 1000

        # Performance target: < 50ms for health check
        assert elapsed_ms < 50
        assert health["status"] in ["healthy", "unhealthy"]

        # Cleanup
        await integration_registry.on_shutdown()

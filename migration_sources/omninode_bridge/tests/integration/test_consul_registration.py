#!/usr/bin/env python3
"""
Integration test for Consul registration in Orchestrator and Reducer nodes.

Tests:
1. Orchestrator registers with Consul on startup
2. Reducer registers with Consul on startup
3. Event metadata includes consul_service_id
4. Deregistration occurs on shutdown
5. Health checks point to correct endpoints

Correlation ID: c5c5ba1d-0642-4aa2-a7a0-086b9592ea67
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Mock omnibase_core if not available
try:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
except ImportError:
    # Use stub from orchestrator node
    from omninode_bridge.nodes.orchestrator.v1_0_0._stubs import ModelONEXContainer


class TestConsulRegistration:
    """Test suite for Consul registration in bridge nodes."""

    @pytest.mark.asyncio
    async def test_orchestrator_consul_registration(self):
        """Test Orchestrator registers with Consul on startup."""
        from omninode_bridge.nodes.orchestrator.v1_0_0.node import (
            NodeBridgeOrchestrator,
        )

        # Create mock container
        container = ModelONEXContainer()
        container.config.from_dict(
            {
                "metadata_stamping_service_url": "http://metadata-stamping:8053",
                "onextree_service_url": "http://onextree:8058",
                "onextree_timeout_ms": 500.0,
                "kafka_broker_url": "localhost:9092",
                "default_namespace": "omninode.bridge",
                "consul_host": "localhost",
                "consul_port": 28500,
                "consul_enable_registration": True,
                "service_port": 8060,
                "service_host": "localhost",
                "health_check_mode": False,
            }
        )

        # Mock Consul client
        with patch("consul.Consul") as mock_consul:
            mock_consul_instance = MagicMock()
            mock_consul.return_value = mock_consul_instance

            # Create orchestrator node
            orchestrator = NodeBridgeOrchestrator(container)

            # Verify registration was called
            mock_consul_instance.agent.service.register.assert_called_once()

            # Verify service_id was stored
            assert hasattr(orchestrator, "_consul_service_id")
            assert orchestrator._consul_service_id.startswith(
                "omninode-bridge-orchestrator-"
            )

            # Verify registration parameters
            call_args = mock_consul_instance.agent.service.register.call_args
            assert call_args.kwargs["name"] == "omninode-bridge-orchestrator"
            assert call_args.kwargs["port"] == 8060
            assert "orchestrator" in call_args.kwargs["tags"]
            assert call_args.kwargs["http"] == "http://localhost:8060/health"

    @pytest.mark.asyncio
    async def test_reducer_consul_registration(self):
        """Test Reducer registers with Consul on startup."""
        from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

        # Create mock container
        container = ModelONEXContainer()
        container.config.from_dict(
            {
                "kafka_broker_url": "localhost:9092",
                "default_namespace": "omninode.bridge",
                "consul_host": "localhost",
                "consul_port": 28500,
                "consul_enable_registration": True,
                "service_port": 8061,
                "service_host": "localhost",
                "health_check_mode": False,
            }
        )

        # Mock Consul client
        with patch("consul.Consul") as mock_consul:
            mock_consul_instance = MagicMock()
            mock_consul.return_value = mock_consul_instance

            # Create reducer node
            reducer = NodeBridgeReducer(container)

            # Verify registration was called
            mock_consul_instance.agent.service.register.assert_called_once()

            # Verify service_id was stored
            assert hasattr(reducer, "_consul_service_id")
            assert reducer._consul_service_id.startswith("omninode-bridge-reducer-")

            # Verify registration parameters
            call_args = mock_consul_instance.agent.service.register.call_args
            assert call_args.kwargs["name"] == "omninode-bridge-reducer"
            assert call_args.kwargs["port"] == 8061
            assert "reducer" in call_args.kwargs["tags"]
            assert call_args.kwargs["http"] == "http://localhost:8061/health"

    @pytest.mark.asyncio
    async def test_orchestrator_event_metadata_includes_consul_service_id(self):
        """Test Orchestrator includes consul_service_id in Kafka event metadata."""
        from omninode_bridge.nodes.orchestrator.v1_0_0.node import (
            NodeBridgeOrchestrator,
        )

        # Create mock container
        container = ModelONEXContainer()
        container.config.from_dict(
            {
                "metadata_stamping_service_url": "http://metadata-stamping:8053",
                "onextree_service_url": "http://onextree:8058",
                "onextree_timeout_ms": 500.0,
                "kafka_broker_url": "localhost:9092",
                "default_namespace": "omninode.bridge",
                "consul_host": "localhost",
                "consul_port": 28500,
                "consul_enable_registration": True,
                "service_port": 8060,
                "service_host": "localhost",
                "health_check_mode": False,
            }
        )

        # Mock Consul and Kafka
        with patch("consul.Consul"):
            orchestrator = NodeBridgeOrchestrator(container)

            # Mock Kafka client
            mock_kafka_client = AsyncMock()
            mock_kafka_client.is_connected = True
            mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
            orchestrator.kafka_client = mock_kafka_client

            # Trigger event publishing
            from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_event import (
                EnumWorkflowEvent,
            )

            await orchestrator._publish_event(
                EnumWorkflowEvent.WORKFLOW_STARTED,
                {
                    "workflow_id": str(uuid4()),
                    "timestamp": "2025-01-01T00:00:00Z",
                },
            )

            # Verify Kafka publish was called with consul_service_id in metadata
            mock_kafka_client.publish_with_envelope.assert_called_once()
            call_args = mock_kafka_client.publish_with_envelope.call_args
            metadata = call_args.kwargs["metadata"]

            assert "consul_service_id" in metadata
            assert metadata["consul_service_id"].startswith(
                "omninode-bridge-orchestrator-"
            )

    @pytest.mark.asyncio
    async def test_reducer_event_metadata_includes_consul_service_id(self):
        """Test Reducer includes consul_service_id in Kafka event metadata."""
        from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

        # Create mock container
        container = ModelONEXContainer()
        container.config.from_dict(
            {
                "kafka_broker_url": "localhost:9092",
                "default_namespace": "omninode.bridge",
                "consul_host": "localhost",
                "consul_port": 28500,
                "consul_enable_registration": True,
                "service_port": 8061,
                "service_host": "localhost",
                "health_check_mode": False,
            }
        )

        # Mock Consul and Kafka
        with patch("consul.Consul"):
            reducer = NodeBridgeReducer(container)

            # Mock Kafka client
            mock_kafka_client = AsyncMock()
            mock_kafka_client.is_connected = True
            mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
            reducer.kafka_client = mock_kafka_client

            # Trigger event publishing
            from omninode_bridge.nodes.reducer.v1_0_0.models.enum_reducer_event import (
                EnumReducerEvent,
            )

            await reducer._publish_event(
                EnumReducerEvent.AGGREGATION_STARTED,
                {
                    "aggregation_id": str(uuid4()),
                    "timestamp": "2025-01-01T00:00:00Z",
                },
            )

            # Verify Kafka publish was called with consul_service_id in metadata
            mock_kafka_client.publish_with_envelope.assert_called_once()
            call_args = mock_kafka_client.publish_with_envelope.call_args
            metadata = call_args.kwargs["metadata"]

            assert "consul_service_id" in metadata
            assert metadata["consul_service_id"].startswith("omninode-bridge-reducer-")

    @pytest.mark.asyncio
    async def test_orchestrator_deregisters_on_shutdown(self):
        """Test Orchestrator deregisters from Consul on shutdown."""
        from omninode_bridge.nodes.orchestrator.v1_0_0.node import (
            NodeBridgeOrchestrator,
        )

        # Create mock container
        container = ModelONEXContainer()
        container.config.from_dict(
            {
                "metadata_stamping_service_url": "http://metadata-stamping:8053",
                "onextree_service_url": "http://onextree:8058",
                "onextree_timeout_ms": 500.0,
                "kafka_broker_url": "localhost:9092",
                "default_namespace": "omninode.bridge",
                "consul_host": "localhost",
                "consul_port": 28500,
                "consul_enable_registration": True,
                "service_port": 8060,
                "service_host": "localhost",
                "health_check_mode": False,
            }
        )

        # Mock Consul
        with patch("consul.Consul") as mock_consul:
            mock_consul_instance = MagicMock()
            mock_consul.return_value = mock_consul_instance

            orchestrator = NodeBridgeOrchestrator(container)
            service_id = orchestrator._consul_service_id

            # Call shutdown
            await orchestrator.shutdown()

            # Verify deregistration was called
            mock_consul_instance.agent.service.deregister.assert_called_once_with(
                service_id
            )

    @pytest.mark.asyncio
    async def test_reducer_deregisters_on_shutdown(self):
        """Test Reducer deregisters from Consul on shutdown."""
        from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

        # Create mock container
        container = ModelONEXContainer()
        container.config.from_dict(
            {
                "kafka_broker_url": "localhost:9092",
                "default_namespace": "omninode.bridge",
                "consul_host": "localhost",
                "consul_port": 28500,
                "consul_enable_registration": True,
                "service_port": 8061,
                "service_host": "localhost",
                "health_check_mode": False,
            }
        )

        # Mock Consul
        with patch("consul.Consul") as mock_consul:
            mock_consul_instance = MagicMock()
            mock_consul.return_value = mock_consul_instance

            reducer = NodeBridgeReducer(container)
            service_id = reducer._consul_service_id

            # Call shutdown
            await reducer.shutdown()

            # Verify deregistration was called
            mock_consul_instance.agent.service.deregister.assert_called_once_with(
                service_id
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

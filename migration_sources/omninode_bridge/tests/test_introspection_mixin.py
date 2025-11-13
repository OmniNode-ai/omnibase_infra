#!/usr/bin/env python3
"""
Comprehensive tests for IntrospectionMixin.

Tests cover:
1. Initialization and configuration
2. Capability extraction from different node types
3. Endpoint discovery and URL generation
4. Introspection broadcasting to Kafka
5. Heartbeat broadcasting
6. Registry request listening (stub)
7. Integration with HealthCheckMixin
8. Error handling and edge cases

ONEX v2.0 Compliance:
- OnexEnvelopeV1 integration validation
- Kafka topic routing validation
- Event correlation ID tracking
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

# Test imports
from omninode_bridge.nodes.mixins.introspection_mixin import IntrospectionMixin

# Test Fixtures
# =============


class MockContainer:
    """Mock ModelONEXContainer for testing."""

    def __init__(self, config=None):
        self.name = "test_container"
        self.version = "1.0.0"
        self.config = config or {}
        self._services = {}

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def get_service(self, service_name: str):
        return self._services.get(service_name)

    def register_service(self, service_name: str, service):
        self._services[service_name] = service


class MockOrchestratorNode(IntrospectionMixin):
    """Mock orchestrator node for testing."""

    def __init__(self, container):
        self.container = container
        self.node_id = str(uuid4())
        self.node_type = "orchestrator"

        # Orchestrator-specific attributes
        self.metadata_stamping_service_url = "http://metadata-stamping:8053"
        self.onextree_service_url = "http://onextree:8080"
        self.kafka_broker_url = "localhost:9092"
        self.workflow_fsm_states = {}
        self.stamping_metrics = {}

        # Add missing attributes for introspection
        self._startup_time = time.time()
        self._registry_listener_task = None

        # Initialize introspection
        super().__init__()

    async def execute_orchestration(self, contract):
        """Mock execute_orchestration method for capability detection."""
        return {"result": "mock_orchestration_result"}


class MockReducerNode(IntrospectionMixin):
    """Mock reducer node for testing."""

    def __init__(self, container):
        self.container = container
        self.node_id = str(uuid4())
        self.node_type = "reducer"

        # Reducer-specific attributes
        self.aggregation_state = {}
        self.namespace_groups = {}

        # Initialize introspection
        super().__init__()

    async def execute_reduction(self, contract):
        """Mock execute_reduction method for capability detection."""
        return {"result": "mock_reduction_result"}


@pytest.fixture
def mock_container():
    """Create mock container with test configuration."""
    return MockContainer(
        config={
            "api_port": 8053,
            "metrics_port": 9090,
            "environment": "development",
            "kafka_broker_url": "localhost:9092",
        }
    )


@pytest.fixture
def orchestrator_node(mock_container):
    """Create mock orchestrator node."""
    return MockOrchestratorNode(mock_container)


@pytest.fixture
def reducer_node(mock_container):
    """Create mock reducer node."""
    return MockReducerNode(mock_container)


# Test Class: Initialization
# ==========================


class TestIntrospectionInitialization:
    """Tests for mixin initialization."""

    def test_initialization_attributes(self, orchestrator_node):
        """Test that initialization creates required attributes."""
        assert hasattr(orchestrator_node, "_introspection_cache")
        assert hasattr(orchestrator_node, "_cache_timestamps")
        assert hasattr(orchestrator_node, "_cache_ttl_seconds")
        assert hasattr(orchestrator_node, "_cached_node_type")

        # Check initial values
        assert isinstance(orchestrator_node._introspection_cache, dict)
        assert isinstance(orchestrator_node._cache_timestamps, dict)
        assert orchestrator_node._cache_ttl_seconds == 300
        assert orchestrator_node._cached_node_type is None

    def test_initialization_completes(self, orchestrator_node):
        """Test that initialization completes successfully."""
        # The mixin should initialize without errors
        assert orchestrator_node is not None
        assert hasattr(orchestrator_node, "get_introspection_data")

    @pytest.mark.asyncio
    async def test_get_introspection_data_basic(self, orchestrator_node):
        """Test basic get_introspection_data functionality."""
        introspection_data = await orchestrator_node.get_introspection_data()

        # Check basic structure
        assert isinstance(introspection_data, dict)
        assert "node_id" in introspection_data
        assert "node_name" in introspection_data
        assert "node_type" in introspection_data
        assert "capabilities" in introspection_data


# Test Class: Capability Extraction
# =================================


class TestCapabilityExtraction:
    """Tests for capability extraction from node metadata."""

    @pytest.mark.asyncio
    async def test_get_orchestrator_capabilities(self, orchestrator_node):
        """Test capability extraction for orchestrator node."""
        capabilities = await orchestrator_node.get_capabilities()

        # Check basic capabilities
        assert "node_type" in capabilities
        assert capabilities["node_type"] == "orchestrator"
        assert "supported_operations" in capabilities
        assert isinstance(capabilities["supported_operations"], list)

        # Check orchestrator-specific operations
        ops = capabilities["supported_operations"]
        assert "coordinate_workflow" in ops
        assert "manage_state" in ops
        assert "route_operations" in ops

    @pytest.mark.asyncio
    async def test_get_reducer_capabilities(self, reducer_node):
        """Test capability extraction for reducer node."""
        capabilities = await reducer_node.get_capabilities()

        # Check basic capabilities
        assert "node_type" in capabilities
        assert capabilities["node_type"] == "reducer"
        assert "supported_operations" in capabilities
        assert isinstance(capabilities["supported_operations"], list)

        # Check reducer-specific operations
        ops = capabilities["supported_operations"]
        assert "execute_reduction" in ops
        assert "aggregate_data" in ops
        assert "reduce_results" in ops

    @pytest.mark.asyncio
    async def test_extract_fsm_states(self, orchestrator_node):
        """Test FSM state extraction for orchestrator."""
        # Add some FSM states
        from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
            EnumWorkflowState,
        )

        orchestrator_node.workflow_fsm_states = {
            "workflow-1": EnumWorkflowState.PROCESSING,
            "workflow-2": EnumWorkflowState.COMPLETED,
        }

        capabilities = await orchestrator_node.get_capabilities()

        assert "fsm_states" in capabilities
        fsm_data = capabilities["fsm_states"]
        assert "supported_states" in fsm_data
        assert "active_workflows" in fsm_data
        assert fsm_data["active_workflows"] == 2

    @pytest.mark.asyncio
    async def test_extract_performance_metrics(self, orchestrator_node):
        """Test performance metrics extraction."""
        # Add metrics
        orchestrator_node.stamping_metrics = {
            "hash_generation": {"total_operations": 100},
            "stamp_creation": {"total_operations": 95},
        }

        capabilities = await orchestrator_node.get_capabilities()

        assert "performance" in capabilities
        perf = capabilities["performance"]
        assert perf["operations_tracked"] == 2
        assert perf["metrics_available"] is True

    @pytest.mark.asyncio
    @patch("omninode_bridge.nodes.mixins.introspection_mixin.PSUTIL_AVAILABLE", True)
    @patch("omninode_bridge.nodes.mixins.introspection_mixin.psutil")
    async def test_extract_resource_limits(self, mock_psutil, orchestrator_node):
        """Test resource limits extraction (with psutil available)."""
        # Mock psutil responses
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = Mock(available=16 * 1024**3)  # 16 GB
        mock_psutil.Process.return_value = Mock(
            memory_info=Mock(return_value=Mock(rss=512 * 1024**2))  # 512 MB
        )

        capabilities = await orchestrator_node.get_capabilities()

        assert "resource_limits" in capabilities
        resources = capabilities["resource_limits"]
        assert resources["cpu_cores_available"] == 8
        assert resources["memory_available_gb"] == 16.0
        assert resources["current_memory_usage_mb"] == 512.0

    @pytest.mark.asyncio
    async def test_extract_service_integration(self, orchestrator_node):
        """Test service integration URLs in capabilities."""
        capabilities = await orchestrator_node.get_capabilities()

        assert "service_integration" in capabilities
        services = capabilities["service_integration"]
        assert "metadata_stamping" in services
        assert services["metadata_stamping"] == "http://metadata-stamping:8053"
        assert "onextree" in services
        assert services["onextree"] == "http://onextree:8080"


# Test Class: Endpoint Discovery
# ==============================


class TestEndpointDiscovery:
    """Tests for endpoint discovery and URL generation."""

    @pytest.mark.asyncio
    async def test_get_orchestrator_endpoints(self, orchestrator_node):
        """Test endpoint discovery for orchestrator."""
        endpoints = await orchestrator_node.get_endpoints()

        # Check required endpoints
        assert "health" in endpoints
        assert "api" in endpoints
        assert "metrics" in endpoints

        # Check orchestrator-specific endpoint
        assert "orchestration" in endpoints

        # Verify URL format
        assert endpoints["health"].startswith("http://")
        assert ":8053" in endpoints["health"]
        assert "/health" in endpoints["health"]

    @pytest.mark.asyncio
    async def test_get_reducer_endpoints(self, reducer_node):
        """Test endpoint discovery for reducer."""
        endpoints = await reducer_node.get_endpoints()

        # Check required endpoints
        assert "health" in endpoints
        assert "api" in endpoints
        assert "metrics" in endpoints

        # Check reducer-specific endpoint
        assert "aggregation" in endpoints

    @pytest.mark.asyncio
    async def test_endpoint_port_configuration(self, mock_container):
        """Test that endpoint ports are read from container config."""
        # Custom ports
        mock_container.config["api_port"] = 9000
        mock_container.config["metrics_port"] = 9100

        node = MockOrchestratorNode(mock_container)
        endpoints = await node.get_endpoints()

        assert ":9000" in endpoints["health"]
        assert ":9000" in endpoints["api"]
        assert ":9100" in endpoints["metrics"]


# Test Class: Introspection Broadcasting
# ======================================


class TestIntrospectionBroadcasting:
    """Tests for introspection event broadcasting."""

    @pytest.mark.asyncio
    async def test_publish_introspection_success(self, orchestrator_node):
        """Test successful introspection broadcasting."""
        # Mock Kafka publishing
        with patch.object(
            orchestrator_node, "_publish_to_kafka", new_callable=AsyncMock
        ) as mock_kafka:
            mock_kafka.return_value = True
            result = await orchestrator_node.publish_introspection(reason="startup")

            assert result is True
            assert mock_kafka.called
            assert orchestrator_node._last_introspection_broadcast is not None

    @pytest.mark.asyncio
    async def test_publish_introspection_with_correlation_id(self, orchestrator_node):
        """Test introspection broadcasting with correlation ID."""
        correlation_id = uuid4()

        with patch.object(
            orchestrator_node, "_publish_to_kafka", new_callable=AsyncMock
        ) as mock_kafka:
            mock_kafka.return_value = True
            result = await orchestrator_node.publish_introspection(
                reason="registry_request", correlation_id=correlation_id
            )

            assert result is True
            # Check that correlation_id was passed to envelope
            call_args = mock_kafka.call_args
            envelope = call_args[0][0]
            assert envelope.correlation_id == str(correlation_id)

    @pytest.mark.asyncio
    async def test_publish_introspection_caching(self, orchestrator_node):
        """Test that capabilities are cached between broadcasts."""
        # First broadcast - should extract capabilities
        await orchestrator_node.publish_introspection(reason="startup")
        assert orchestrator_node._cached_capabilities is not None

        # Modify underlying data
        orchestrator_node.workflow_fsm_states["test"] = "value"

        # Second broadcast without force_refresh - should use cache
        await orchestrator_node.publish_introspection(
            reason="periodic", force_refresh=False
        )
        # Cache should still be the old capabilities (without new workflow state)

    @pytest.mark.asyncio
    async def test_publish_introspection_force_refresh(self, orchestrator_node):
        """Test that force_refresh updates cached capabilities."""
        # First broadcast
        await orchestrator_node.publish_introspection(reason="startup")
        old_cache = orchestrator_node._cached_capabilities.copy()

        # Modify underlying data
        orchestrator_node.workflow_fsm_states["test"] = "value"

        # Broadcast with force_refresh
        await orchestrator_node.publish_introspection(
            reason="manual", force_refresh=True
        )
        new_cache = orchestrator_node._cached_capabilities

        # Cache should be updated
        assert new_cache != old_cache

    @pytest.mark.asyncio
    async def test_publish_introspection_error_handling(self, orchestrator_node):
        """Test error handling during introspection broadcasting."""
        # Mock Kafka publishing to raise exception
        with patch.object(
            orchestrator_node,
            "_publish_to_kafka",
            new_callable=AsyncMock,
            side_effect=Exception("Kafka error"),
        ):
            result = await orchestrator_node.publish_introspection(reason="test")

            # Should return False on error, but not raise
            assert result is False


# Test Class: Heartbeat Broadcasting
# ==================================


class TestHeartbeatBroadcasting:
    """Tests for heartbeat event broadcasting."""

    @pytest.mark.asyncio
    async def test_publish_heartbeat_success(self, orchestrator_node):
        """Test successful heartbeat broadcasting."""
        with patch.object(
            orchestrator_node, "_publish_to_kafka", new_callable=AsyncMock
        ) as mock_kafka:
            mock_kafka.return_value = True
            result = await orchestrator_node._publish_heartbeat()

            assert result is True
            assert mock_kafka.called

            # Check envelope structure
            call_args = mock_kafka.call_args
            envelope = call_args[0][0]
            assert envelope.event_type == "node_heartbeat"  # Enum value is lowercase

    @pytest.mark.asyncio
    async def test_heartbeat_includes_uptime(self, orchestrator_node):
        """Test that heartbeat includes uptime calculation."""
        # Wait a bit to ensure uptime is non-zero
        await asyncio.sleep(0.1)

        with patch.object(
            orchestrator_node, "_publish_to_kafka", new_callable=AsyncMock
        ) as mock_kafka:
            await orchestrator_node._publish_heartbeat()

            call_args = mock_kafka.call_args
            envelope = call_args[0][0]
            payload = envelope.payload

            # Uptime should be non-negative (0 is valid immediately after startup)
            assert payload["uptime_seconds"] >= 0
            assert isinstance(payload["uptime_seconds"], int)

    @pytest.mark.asyncio
    async def test_heartbeat_includes_active_operations(self, orchestrator_node):
        """Test that heartbeat includes active operations count."""
        # Add some active workflows
        from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
            EnumWorkflowState,
        )

        orchestrator_node.workflow_fsm_states = {
            "workflow-1": EnumWorkflowState.PROCESSING,
            "workflow-2": EnumWorkflowState.PROCESSING,
        }

        with patch.object(
            orchestrator_node, "_publish_to_kafka", new_callable=AsyncMock
        ) as mock_kafka:
            await orchestrator_node._publish_heartbeat()

            call_args = mock_kafka.call_args
            envelope = call_args[0][0]
            payload = envelope.payload

            assert payload["active_operations"] == 2

    @pytest.mark.asyncio
    async def test_heartbeat_loop_cancellation(self, orchestrator_node):
        """Test that heartbeat loop can be cancelled."""
        # Start heartbeat loop with very short interval
        task = asyncio.create_task(orchestrator_node._heartbeat_loop(0.1))

        # Wait a bit
        await asyncio.sleep(0.2)

        # Cancel task
        task.cancel()

        # Should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await task


# Test Class: Background Tasks
# ============================


class TestBackgroundTasks:
    """Tests for background introspection tasks."""

    @pytest.mark.asyncio
    async def test_start_introspection_tasks(self, orchestrator_node):
        """Test starting background introspection tasks."""
        with (
            patch.object(
                orchestrator_node, "_heartbeat_loop", new_callable=AsyncMock
            ) as mock_heartbeat,
            patch.object(
                orchestrator_node, "_registry_listener_loop", new_callable=AsyncMock
            ) as mock_registry,
        ):
            await orchestrator_node.start_introspection_tasks(
                enable_heartbeat=True,
                heartbeat_interval_seconds=30,
                enable_registry_listener=True,
            )

            # Check tasks were created
            assert orchestrator_node._heartbeat_task is not None
            assert orchestrator_node._registry_listener_task is not None

            # Cleanup
            await orchestrator_node.stop_introspection_tasks()

    @pytest.mark.asyncio
    async def test_stop_introspection_tasks(self, orchestrator_node):
        """Test stopping background introspection tasks."""
        # Start tasks
        await orchestrator_node.start_introspection_tasks(
            enable_heartbeat=True, enable_registry_listener=True
        )

        # Stop tasks
        await orchestrator_node.stop_introspection_tasks()

        # Tasks should be done (cancelled or completed)
        # Heartbeat should be cancelled (runs forever)
        assert orchestrator_node._heartbeat_task.cancelled()
        # Registry listener may complete early if kafka_client not available, or be cancelled
        assert (
            orchestrator_node._registry_listener_task.cancelled()
            or orchestrator_node._registry_listener_task.done()
        )

    @pytest.mark.asyncio
    async def test_start_heartbeat_only(self, orchestrator_node):
        """Test starting only heartbeat task."""
        await orchestrator_node.start_introspection_tasks(
            enable_heartbeat=True, enable_registry_listener=False
        )

        assert orchestrator_node._heartbeat_task is not None
        assert orchestrator_node._registry_listener_task is None

        # Cleanup
        await orchestrator_node.stop_introspection_tasks()


# Test Class: Kafka Integration
# =============================


class TestKafkaIntegration:
    """Tests for Kafka producer integration."""

    @pytest.mark.asyncio
    async def test_publish_to_kafka_with_producer(self, orchestrator_node):
        """Test publishing to Kafka when producer is available."""
        # Mock Kafka producer - set as attribute (not service)
        mock_producer = Mock()
        orchestrator_node.kafka_producer = mock_producer

        # Mock envelope
        mock_envelope = Mock()
        mock_envelope.to_kafka_topic.return_value = "test-topic"
        mock_envelope.get_kafka_key.return_value = "test-key"
        mock_envelope.model_dump_json.return_value = '{"test": "data"}'
        mock_envelope.event_type = "NODE_INTROSPECTION"

        # Publish
        result = await orchestrator_node._publish_to_kafka(mock_envelope)

        # Verify publish was successful
        assert result is True
        # Verify envelope was stored in _kafka_messages
        assert hasattr(orchestrator_node, "_kafka_messages")
        assert len(orchestrator_node._kafka_messages) == 1
        assert orchestrator_node._kafka_messages[0] == mock_envelope

    @pytest.mark.asyncio
    async def test_publish_to_kafka_without_producer(self, orchestrator_node):
        """Test publishing to Kafka when producer is not available."""
        # Ensure no producer is registered
        assert orchestrator_node.container.get_service("kafka_producer") is None

        # Mock envelope
        mock_envelope = Mock()
        mock_envelope.to_kafka_topic.return_value = "test-topic"
        mock_envelope.get_kafka_key.return_value = "test-key"
        mock_envelope.event_type = "NODE_INTROSPECTION"

        # Should not raise, just log
        await orchestrator_node._publish_to_kafka(mock_envelope)


# Test Class: Metadata Helpers
# ============================


class TestMetadataHelpers:
    """Tests for internal metadata helper methods."""

    def test_get_node_type_from_attribute(self, orchestrator_node):
        """Test getting node type from attribute."""
        node_type = orchestrator_node._get_node_type()
        assert node_type == "orchestrator"

    def test_get_node_type_from_class_name(self, mock_container):
        """Test inferring node type from class name."""

        class NodeCustomReducerTest(IntrospectionMixin):
            def __init__(self, container):
                self.container = container
                self.node_id = str(uuid4())
                super().__init__()

        node = NodeCustomReducerTest(mock_container)
        node_type = node._get_node_type()
        assert node_type == "reducer"

    def test_get_supported_operations_orchestrator(self, orchestrator_node):
        """Test getting supported operations for orchestrator."""
        ops = orchestrator_node._get_supported_operations_cached("orchestrator")

        assert "coordinate_workflow" in ops
        assert "manage_state" in ops
        assert "route_operations" in ops

    def test_get_supported_operations_reducer(self, reducer_node):
        """Test getting supported operations for reducer."""
        ops = reducer_node._get_supported_operations_cached("reducer")

        assert "aggregate_data" in ops
        assert "reduce_results" in ops
        assert "manage_state" in ops

    @pytest.mark.asyncio
    async def test_get_introspection_metadata(self, orchestrator_node):
        """Test getting introspection metadata."""
        metadata = await orchestrator_node.get_introspection_data()

        # Check new API fields
        assert "node_id" in metadata
        assert "node_name" in metadata
        assert "node_type" in metadata
        assert "node_version" in metadata
        assert "capabilities" in metadata
        assert "endpoints" in metadata
        assert "configuration" in metadata
        assert "current_state" in metadata
        assert "introspection_timestamp" in metadata
        assert "introspection_version" in metadata

    def test_get_environment_from_container(self, orchestrator_node):
        """Test getting environment from container."""
        env = orchestrator_node._get_environment()
        assert env == "development"

    def test_get_environment_default(self, mock_container):
        """Test default environment when not configured."""
        # Remove environment from config
        mock_container.config.pop("environment", None)

        node = MockOrchestratorNode(mock_container)
        env = node._get_environment()
        assert env == "development"

    def test_get_environment_with_dependency_injector_provider(self):
        """Test that ConfigurationOption providers are properly resolved to strings."""
        from dependency_injector import containers, providers

        class TestContainer(containers.DeclarativeContainer):
            config = providers.Configuration()

        # Create container with dependency_injector ConfigurationOption
        container = TestContainer()
        container.config.environment.from_value("production")

        # Create node with this container
        node = MockOrchestratorNode(container)

        # Get environment - should resolve ConfigurationOption to string
        env = node._get_environment()
        assert isinstance(env, str), f"Expected str, got {type(env)}"
        assert env == "production"

        # Verify it can be used in f-strings (the original bug scenario)
        topic_name = f"{env}.omninode_bridge.onex.evt.registry-request-introspection.v1"
        assert (
            topic_name
            == "production.omninode_bridge.onex.evt.registry-request-introspection.v1"
        )
        assert "ConfigurationOption" not in topic_name


# Test Class: Edge Cases
# ======================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_node_without_container(self):
        """Test behavior when node doesn't have container attribute."""

        class MinimalNode(IntrospectionMixin):
            def __init__(self):
                self.node_id = str(uuid4())
                self.node_type = "test"
                super().__init__()

        node = MinimalNode()

        # Should not raise, use defaults
        capabilities = await node.get_capabilities()
        assert "node_type" in capabilities

    @pytest.mark.asyncio
    async def test_node_without_node_type(self, mock_container):
        """Test behavior when node doesn't have explicit node_type."""

        class NodeWithoutType(IntrospectionMixin):
            def __init__(self, container):
                self.container = container
                self.node_id = str(uuid4())
                super().__init__()

        node = NodeWithoutType(mock_container)

        # Should infer "unknown" when can't determine type
        node_type = node._get_node_type()
        assert node_type == "unknown"

    @pytest.mark.asyncio
    async def test_publish_introspection_import_error(self, orchestrator_node):
        """Test handling of import errors during introspection."""
        # Mock get_capabilities to fail
        with patch.object(
            orchestrator_node,
            "get_capabilities",
            side_effect=ImportError("Module not found"),
        ):
            result = await orchestrator_node.publish_introspection(reason="test")

            # Should return False but not raise
            assert result is False


# Test Class: Registry Request Listener
# =====================================


class TestRegistryRequestListener:
    """Tests for the registry request Kafka consumer implementation."""

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_no_container(self, mock_container):
        """Test graceful handling when container is not available."""

        class NodeWithoutContainer(IntrospectionMixin):
            def __init__(self):
                self.node_id = str(uuid4())
                super().__init__()

        node = NodeWithoutContainer()

        # Should return early without error
        await node.listen_for_registry_requests()

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_no_kafka_client(
        self, orchestrator_node
    ):
        """Test graceful handling when Kafka client is not available."""
        # Ensure container returns None for kafka_client
        orchestrator_node.container._services.pop("kafka_client", None)

        # Should return early without error
        await orchestrator_node.listen_for_registry_requests()

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_success(self, orchestrator_node):
        """Test successful processing of registry request messages."""
        # Mock Kafka client
        mock_kafka_client = Mock()
        orchestrator_node.container.register_service("kafka_client", mock_kafka_client)

        # Mock message with registry request
        mock_message = Mock()
        mock_envelope_data = {
            "correlation_id": str(uuid4()),
            "payload": {
                "registry_id": "test-registry",
                "reason": "startup_rebroadcast",
                "target_node_types": None,  # All nodes should respond
                "response_timeout_ms": 5000,
                "metadata": {},
            },
        }

        # Create a proper envelope
        from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_registry_request_event import (
            EnumIntrospectionReason,
            ModelRegistryRequestEvent,
        )
        from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
            ModelOnexEnvelopeV1,
        )

        request_event = ModelRegistryRequestEvent(
            registry_id="test-registry",
            reason=EnumIntrospectionReason.STARTUP_REBROADCAST,
            target_node_types=None,
            response_timeout_ms=5000,
            metadata={},
        )

        envelope = ModelOnexEnvelopeV1(
            event_type="registry-request-introspection",
            source_node_id="test-registry",
            payload=request_event.to_dict(),
            correlation_id=uuid4(),
        )

        mock_message.value = envelope.to_bytes()

        # Mock consume_messages to return messages once, then timeout
        call_count = 0

        async def mock_consume(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_message]
            # Cancel after first message
            raise asyncio.CancelledError()

        mock_kafka_client.consume_messages = AsyncMock(side_effect=mock_consume)
        mock_kafka_client.commit_offsets = AsyncMock()

        # Mock publish_introspection
        orchestrator_node.publish_introspection = AsyncMock(return_value=True)

        # Run listener (should process one message then cancel)
        try:
            await orchestrator_node.listen_for_registry_requests()
        except asyncio.CancelledError:
            pass

        # Verify behavior
        assert mock_kafka_client.consume_messages.call_count >= 1
        orchestrator_node.publish_introspection.assert_called_once()

        # Verify publish_introspection called with correct params
        call_args = orchestrator_node.publish_introspection.call_args
        assert call_args[1]["reason"] == "registry_request"
        assert call_args[1]["correlation_id"] == envelope.correlation_id
        assert call_args[1]["force_refresh"] is True

        # Verify offset commit
        mock_kafka_client.commit_offsets.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_filtered_node_type(
        self, orchestrator_node
    ):
        """Test filtering by target_node_types."""
        # Mock Kafka client
        mock_kafka_client = Mock()
        orchestrator_node.container.register_service("kafka_client", mock_kafka_client)

        # Create request targeting only reducers (not orchestrators)
        from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_registry_request_event import (
            EnumIntrospectionReason,
            ModelRegistryRequestEvent,
        )
        from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
            ModelOnexEnvelopeV1,
        )

        request_event = ModelRegistryRequestEvent(
            registry_id="test-registry",
            reason=EnumIntrospectionReason.REFRESH,
            target_node_types=["reducer", "effect"],  # Excludes orchestrator
            response_timeout_ms=5000,
            metadata={},
        )

        envelope = ModelOnexEnvelopeV1(
            event_type="registry-request-introspection",
            source_node_id="test-registry",
            payload=request_event.to_dict(),
            correlation_id=uuid4(),
        )

        mock_message = Mock()
        mock_message.value = envelope.to_bytes()

        # Mock consume_messages to return one filtered message, then cancel
        call_count = 0

        async def mock_consume(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_message]
            raise asyncio.CancelledError()

        mock_kafka_client.consume_messages = AsyncMock(side_effect=mock_consume)
        mock_kafka_client.commit_offsets = AsyncMock()

        # Mock publish_introspection
        orchestrator_node.publish_introspection = AsyncMock(return_value=True)

        # Run listener
        try:
            await orchestrator_node.listen_for_registry_requests()
        except asyncio.CancelledError:
            pass

        # Should NOT call publish_introspection because orchestrator is not in target_node_types
        orchestrator_node.publish_introspection.assert_not_called()

        # Should still commit offsets
        mock_kafka_client.commit_offsets.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_error_handling(self, orchestrator_node):
        """Test error handling for malformed messages."""
        # Mock Kafka client
        mock_kafka_client = Mock()
        orchestrator_node.container.register_service("kafka_client", mock_kafka_client)

        # Create malformed message
        mock_message = Mock()
        mock_message.value = b"invalid_bytes_not_a_valid_envelope"

        # Mock consume_messages to return malformed message, then cancel
        call_count = 0

        async def mock_consume(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_message]
            raise asyncio.CancelledError()

        mock_kafka_client.consume_messages = AsyncMock(side_effect=mock_consume)
        mock_kafka_client.commit_offsets = AsyncMock()

        # Mock publish_introspection (should not be called)
        orchestrator_node.publish_introspection = AsyncMock(return_value=True)

        # Run listener (should handle error gracefully)
        try:
            await orchestrator_node.listen_for_registry_requests()
        except asyncio.CancelledError:
            pass

        # Should NOT call publish_introspection due to deserialization error
        orchestrator_node.publish_introspection.assert_not_called()

        # Should still commit offsets (best-effort)
        mock_kafka_client.commit_offsets.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_timeout_normal(self, orchestrator_node):
        """Test that timeouts are handled as normal (no messages)."""
        # Mock Kafka client
        mock_kafka_client = Mock()
        orchestrator_node.container.register_service("kafka_client", mock_kafka_client)

        # Mock consume_messages to timeout twice, then cancel
        call_count = 0

        async def mock_consume(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise asyncio.TimeoutError()
            raise asyncio.CancelledError()

        mock_kafka_client.consume_messages = AsyncMock(side_effect=mock_consume)

        # Run listener
        try:
            await orchestrator_node.listen_for_registry_requests()
        except asyncio.CancelledError:
            pass

        # Should handle timeouts gracefully
        assert mock_kafka_client.consume_messages.call_count >= 2

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_publish_failure(
        self, orchestrator_node
    ):
        """Test handling when publish_introspection fails."""
        # Mock Kafka client
        mock_kafka_client = Mock()
        orchestrator_node.container.register_service("kafka_client", mock_kafka_client)

        # Create valid request
        from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_registry_request_event import (
            EnumIntrospectionReason,
            ModelRegistryRequestEvent,
        )
        from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
            ModelOnexEnvelopeV1,
        )

        request_event = ModelRegistryRequestEvent(
            registry_id="test-registry",
            reason=EnumIntrospectionReason.STARTUP_REBROADCAST,
            target_node_types=None,
            response_timeout_ms=5000,
            metadata={},
        )

        envelope = ModelOnexEnvelopeV1(
            event_type="registry-request-introspection",
            source_node_id="test-registry",
            payload=request_event.to_dict(),
            correlation_id=uuid4(),
        )

        mock_message = Mock()
        mock_message.value = envelope.to_bytes()

        # Mock consume_messages
        call_count = 0

        async def mock_consume(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_message]
            raise asyncio.CancelledError()

        mock_kafka_client.consume_messages = AsyncMock(side_effect=mock_consume)
        mock_kafka_client.commit_offsets = AsyncMock()

        # Mock publish_introspection to fail
        orchestrator_node.publish_introspection = AsyncMock(return_value=False)

        # Run listener (should handle publish failure gracefully)
        try:
            await orchestrator_node.listen_for_registry_requests()
        except asyncio.CancelledError:
            pass

        # Should attempt to publish
        orchestrator_node.publish_introspection.assert_called_once()

        # Should still commit offsets
        mock_kafka_client.commit_offsets.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_for_registry_requests_metrics_tracking(
        self, orchestrator_node
    ):
        """Test that metrics are tracked during listener operation."""
        # Mock Kafka client
        mock_kafka_client = Mock()
        orchestrator_node.container.register_service("kafka_client", mock_kafka_client)

        # Create multiple valid requests
        from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_registry_request_event import (
            EnumIntrospectionReason,
            ModelRegistryRequestEvent,
        )
        from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
            ModelOnexEnvelopeV1,
        )

        messages = []
        for _ in range(3):
            request_event = ModelRegistryRequestEvent(
                registry_id="test-registry",
                reason=EnumIntrospectionReason.REFRESH,
                target_node_types=None,
                response_timeout_ms=5000,
                metadata={},
            )

            envelope = ModelOnexEnvelopeV1(
                event_type="registry-request-introspection",
                source_node_id="test-registry",
                payload=request_event.to_dict(),
                correlation_id=uuid4(),
            )

            mock_message = Mock()
            mock_message.value = envelope.to_bytes()
            messages.append(mock_message)

        # Mock consume_messages to return 3 messages, then cancel
        call_count = 0

        async def mock_consume(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return messages
            raise asyncio.CancelledError()

        mock_kafka_client.consume_messages = AsyncMock(side_effect=mock_consume)
        mock_kafka_client.commit_offsets = AsyncMock()

        # Mock publish_introspection
        orchestrator_node.publish_introspection = AsyncMock(return_value=True)

        # Run listener
        try:
            await orchestrator_node.listen_for_registry_requests()
        except asyncio.CancelledError:
            pass

        # Verify all 3 messages were processed
        assert orchestrator_node.publish_introspection.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

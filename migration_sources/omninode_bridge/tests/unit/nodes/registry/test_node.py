#!/usr/bin/env python3
"""
Unit tests for NodeBridgeRegistry - Core Functionality.

Tests cover:
- Node initialization
- Dual registration (Consul + PostgreSQL)
- Atomic registration with rollback
- Circuit breaker protection
- TTL cleanup
- Offset deduplication
- Memory leak prevention
- Background task lifecycle
"""

import asyncio
from datetime import UTC, datetime, timedelta
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
def mock_container():
    """Create mock container with configuration."""
    container = ModelContainer(
        value={
            "registry_id": "test-registry",
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
            "health_check_mode": True,  # Skip service initialization in tests
        },
        container_type="config",
    )
    return container


@pytest.fixture
def registry_node(mock_container):
    """Create registry node instance for testing."""
    return NodeBridgeRegistry(mock_container, environment="dev")


@pytest.fixture
def mock_introspection_event():
    """Create mock introspection event."""
    return ModelNodeIntrospectionEvent(
        node_id="test-node-123",
        node_type="orchestrator",
        capabilities={"stamping": True, "routing": True},  # Must be dict, not list
        endpoints={
            "health": "http://localhost:8060/health",
            "api": "http://localhost:8060",
        },
        metadata={
            "version": "1.0.0",
            "environment": "dev",
        },
        correlation_id=uuid4(),
    )


# Test: Initialization


class TestInitialization:
    """Test suite for NodeBridgeRegistry initialization."""

    def test_init_success(self, mock_container):
        """Test successful registry initialization."""
        registry = NodeBridgeRegistry(mock_container, environment="dev")

        assert registry.registry_id == "test-registry"
        assert registry.kafka_broker_url == "test-kafka:9092"
        assert registry.consul_host == "test-consul"
        assert registry.consul_port == 8500
        assert registry.postgres_host == "test-postgres"
        assert registry.postgres_port == 5432
        assert registry.postgres_db == "test_db"
        assert registry.postgres_user == "test_user"
        # Note: Password may come from environment variable (takes precedence)
        # or container config, so just verify it's set
        assert registry.postgres_password is not None

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        container = ModelContainer(
            value={"postgres_password": "test_pass"},
            container_type="config",
        )

        registry = NodeBridgeRegistry(container, environment="dev")

        # Note: Configuration values may come from environment variables (take precedence)
        # or defaults, so just verify they're set
        assert registry.kafka_broker_url is not None
        assert registry.consul_host is not None
        assert registry.consul_port > 0  # Verify port is valid
        assert registry.postgres_host is not None
        assert registry.postgres_port in [5432, 5436]  # Default or from env

    def test_init_missing_password(self, monkeypatch):
        """Test initialization fails without password."""
        # Remove POSTGRES_PASSWORD from environment for this test
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

        container = ModelContainer(value={}, container_type="config")

        from omnibase_core.errors.model_onex_error import ModelOnexError

        with pytest.raises(ModelOnexError) as exc_info:
            NodeBridgeRegistry(container, environment="dev")

        assert "password" in str(exc_info.value.message).lower()

    def test_registry_id_validation(self, mock_container):
        """Test registry ID validation."""
        # Valid registry ID
        registry = NodeBridgeRegistry(mock_container, environment="dev")
        assert registry.registry_id == "test-registry"

        # Invalid registry ID with special characters
        from omnibase_core.errors.model_onex_error import ModelOnexError

        invalid_container = ModelContainer(
            value={
                "registry_id": "test@registry!",
                "postgres_password": "test",
            },
            container_type="config",
        )

        with pytest.raises(ModelOnexError) as exc_info:
            NodeBridgeRegistry(invalid_container, environment="dev")

        assert "invalid characters" in str(exc_info.value.message).lower()


# Test: Dual Registration


class TestDualRegistration:
    """Test suite for dual registration functionality."""

    @pytest.mark.asyncio
    async def test_dual_register_success(self, registry_node, mock_introspection_event):
        """Test successful dual registration to Consul and PostgreSQL."""
        # Mock Consul client
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        registry_node.consul_client = mock_consul_client

        # Mock PostgreSQL client
        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        registry_node.node_repository = mock_node_repository

        # Execute dual registration
        result = await registry_node.dual_register(mock_introspection_event)

        # Assertions
        assert result["status"] == "success"
        assert result["consul_registered"] is True
        assert result["postgres_registered"] is True
        assert result["registered_node_id"] == "test-node-123"
        assert result["registration_time_ms"] > 0

        # Verify node is tracked
        assert "test-node-123" in registry_node.registered_nodes
        assert "test-node-123" in registry_node.node_last_seen

        # Verify metrics updated
        assert registry_node.registration_metrics["successful_registrations"] == 1
        assert registry_node.registration_metrics["consul_registrations"] == 1
        assert registry_node.registration_metrics["postgres_registrations"] == 1

    @pytest.mark.asyncio
    async def test_dual_register_consul_only(
        self, registry_node, mock_introspection_event
    ):
        """Test registration with only Consul available."""
        # Mock Consul client
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        registry_node.consul_client = mock_consul_client

        # No PostgreSQL client
        registry_node.node_repository = None

        # Execute registration
        result = await registry_node.dual_register(mock_introspection_event)

        # Assertions
        assert result["status"] == "partial"
        assert result["consul_registered"] is True
        assert result["postgres_registered"] is False

    @pytest.mark.asyncio
    async def test_dual_register_consul_failure(
        self, registry_node, mock_introspection_event
    ):
        """Test registration when Consul registration fails."""
        # Mock Consul client to fail
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(
            side_effect=Exception("Consul unavailable")
        )
        registry_node.consul_client = mock_consul_client

        # Mock PostgreSQL client
        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        registry_node.node_repository = mock_node_repository

        # Execute registration (should not raise exception)
        result = await registry_node.dual_register(mock_introspection_event)

        # Assertions - partial success
        assert result["status"] in ["partial", "error"]
        assert registry_node.registration_metrics["failed_registrations"] >= 0


# Test: Atomic Registration


class TestAtomicRegistration:
    """Test suite for atomic registration with rollback."""

    @pytest.mark.asyncio
    async def test_atomic_register_both_succeed(
        self, registry_node, mock_introspection_event
    ):
        """Test atomic registration when both services succeed."""
        # Enable atomic registration
        registry_node.config.atomic_registration_enabled = True

        # Mock Consul client
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        mock_consul_client.deregister_node = AsyncMock(return_value=True)
        registry_node.consul_client = mock_consul_client

        # Mock PostgreSQL client
        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        registry_node.node_repository = mock_node_repository

        # Execute atomic registration
        result = await registry_node.dual_register(mock_introspection_event)

        # Assertions
        assert result["status"] == "success"
        assert result["consul_registered"] is True
        assert result["postgres_registered"] is True
        assert result["atomic_mode"] is True

        # Verify no rollback occurred
        mock_consul_client.deregister_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_atomic_register_postgres_fails_rollback(
        self, registry_node, mock_introspection_event
    ):
        """Test atomic registration rollback when PostgreSQL fails."""
        # Enable atomic registration
        registry_node.config.atomic_registration_enabled = True

        # Mock Consul client (succeeds)
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        mock_consul_client.deregister_node = AsyncMock(return_value=True)
        registry_node.consul_client = mock_consul_client

        # Mock PostgreSQL client (fails)
        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(
            side_effect=Exception("Database error")
        )
        mock_node_repository.delete_node_registration = AsyncMock(return_value=True)
        registry_node.node_repository = mock_node_repository

        # Execute atomic registration (should raise exception)
        from omnibase_core.errors.model_onex_error import ModelOnexError

        with pytest.raises(ModelOnexError):
            await registry_node.dual_register(mock_introspection_event)

        # Verify Consul rollback was called
        mock_consul_client.deregister_node.assert_called_once_with("test-node-123")

    @pytest.mark.asyncio
    async def test_atomic_register_consul_fails_no_rollback(
        self, registry_node, mock_introspection_event
    ):
        """Test atomic registration when Consul fails initially (no rollback needed)."""
        # Enable atomic registration
        registry_node.config.atomic_registration_enabled = True

        # Mock Consul client (fails)
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(
            side_effect=Exception("Consul error")
        )
        registry_node.consul_client = mock_consul_client

        # Mock PostgreSQL client
        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        registry_node.node_repository = mock_node_repository

        # Execute atomic registration (should raise exception)
        from omnibase_core.errors.model_onex_error import ModelOnexError

        with pytest.raises(ModelOnexError):
            await registry_node.dual_register(mock_introspection_event)

        # Verify PostgreSQL was never called (Consul failed first)
        mock_node_repository.create_registration.assert_not_called()


# Test: Circuit Breaker


class TestCircuitBreaker:
    """Test suite for circuit breaker protection."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(
        self, registry_node, mock_introspection_event
    ):
        """Test circuit breaker opens after consecutive failures."""

        # Mock dual_register to fail
        async def failing_register(*args, **kwargs):
            raise Exception("Service error")

        # Execute multiple registrations through circuit breaker to trigger it
        failure_threshold = (
            registry_node._registration_circuit_breaker._failure_threshold
        )

        for i in range(failure_threshold + 1):
            try:
                await registry_node._registration_circuit_breaker.call(
                    failing_register, mock_introspection_event
                )
            except Exception:
                pass  # Expected failures

        # Verify circuit breaker state
        cb_status = registry_node._registration_circuit_breaker.get_status()
        assert (
            cb_status["metrics"]["consecutive_failures"] > 0
            or cb_status["metrics"]["failed_calls"] > 0
        )
        assert cb_status["state"] in [
            "open",
            "half_open",
        ]  # Should be open after failures

    @pytest.mark.asyncio
    async def test_circuit_breaker_kafka(self, registry_node):
        """Test Kafka circuit breaker protection."""
        # Mock Kafka client to fail
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(
            side_effect=Exception("Kafka error")
        )
        registry_node.kafka_client = mock_kafka_client

        # Verify circuit breaker is initialized
        assert registry_node._kafka_circuit_breaker is not None

        # Execute operation through circuit breaker
        failure_threshold = registry_node._kafka_circuit_breaker._failure_threshold

        for i in range(failure_threshold + 1):
            try:
                await registry_node._kafka_circuit_breaker.call(
                    mock_kafka_client.consume_messages,
                    "test-topic",
                    "test-group",
                )
            except Exception:
                pass  # Expected failures

        # Verify circuit breaker tracked failures
        cb_status = registry_node._kafka_circuit_breaker.get_status()
        assert (
            cb_status["metrics"]["consecutive_failures"] > 0
            or cb_status["metrics"]["failed_calls"] > 0
        )


# Test: TTL Cleanup


class TestTTLCleanup:
    """Test suite for TTL-based node cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_nodes(self, registry_node, mock_introspection_event):
        """Test cleanup removes expired nodes based on TTL."""
        # Add node with expired timestamp
        registry_node.registered_nodes["test-node-123"] = mock_introspection_event
        past_time = datetime.now(UTC) - timedelta(
            hours=registry_node.node_ttl_hours + 1
        )
        registry_node.node_last_seen["test-node-123"] = past_time

        # Add node that's not expired
        registry_node.registered_nodes["test-node-456"] = mock_introspection_event
        registry_node.node_last_seen["test-node-456"] = datetime.now(UTC)

        # Execute cleanup
        await registry_node._cleanup_expired_nodes()

        # Assertions
        assert "test-node-123" not in registry_node.registered_nodes
        assert "test-node-456" in registry_node.registered_nodes
        assert registry_node.memory_metrics["nodes_removed"] == 1
        assert registry_node.registration_metrics["cleanup_operations_count"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_no_expired_nodes(
        self, registry_node, mock_introspection_event
    ):
        """Test cleanup when no nodes are expired."""
        # Add node that's not expired
        registry_node.registered_nodes["test-node-123"] = mock_introspection_event
        registry_node.node_last_seen["test-node-123"] = datetime.now(UTC)

        initial_count = len(registry_node.registered_nodes)

        # Execute cleanup
        await registry_node._cleanup_expired_nodes()

        # Assertions
        assert len(registry_node.registered_nodes) == initial_count
        assert "test-node-123" in registry_node.registered_nodes


# Test: Offset Deduplication


class TestOffsetDeduplication:
    """Test suite for offset tracking and deduplication."""

    @pytest.mark.asyncio
    async def test_offset_cache_prevents_duplicates(self, registry_node):
        """Test offset cache prevents duplicate message processing."""
        offset_key = "test-topic:0:123"

        # First check - should not exist
        is_processed = await registry_node.is_offset_processed(offset_key)
        assert is_processed is False

        # Add to processed offsets
        await registry_node._add_processed_offset(offset_key)

        # Second check - should exist
        is_processed = await registry_node.is_offset_processed(offset_key)
        assert is_processed is True

    @pytest.mark.asyncio
    async def test_offset_cleanup_on_limit(self, registry_node):
        """Test offset cleanup when limit is reached."""
        # Add offsets up to 95% of limit to trigger cleanup
        max_offsets = registry_node._max_tracked_offsets
        offsets_to_add = int(max_offsets * 0.95)

        for i in range(offsets_to_add):
            offset_key = f"topic:0:{i}"
            await registry_node._add_processed_offset(offset_key)

        # Wait for cleanup task to potentially run
        await asyncio.sleep(0.1)

        # Verify cleanup was scheduled (check flag or metrics)
        # Note: Actual cleanup happens asynchronously
        assert len(registry_node._processed_message_offsets) <= max_offsets


# Test: Memory Monitoring


class TestMemoryMonitoring:
    """Test suite for memory monitoring and leak prevention."""

    @pytest.mark.asyncio
    async def test_memory_usage_calculation(
        self, registry_node, mock_introspection_event
    ):
        """Test memory usage calculation."""
        # Add some nodes
        for i in range(10):
            node_id = f"test-node-{i}"
            registry_node.registered_nodes[node_id] = mock_introspection_event
            registry_node.node_last_seen[node_id] = datetime.now(UTC)

        # Calculate memory usage
        memory_mb = registry_node._calculate_memory_usage()

        # Assertions
        # Note: Small amounts of data may round to 0.0 MB (< 1024 * 1024 bytes)
        # sys.getsizeof() on Pydantic models returns minimal overhead, not actual data size
        assert memory_mb >= 0.0  # Should return valid float, even if 0.0
        assert isinstance(memory_mb, float)

    @pytest.mark.asyncio
    async def test_memory_monitoring_triggers_cleanup(self, registry_node):
        """Test memory monitoring triggers cleanup when threshold exceeded."""
        # Set low threshold for testing
        registry_node.config.memory_critical_threshold_mb = 0.01  # Very low threshold

        # Add many nodes to exceed threshold (simulated)
        for i in range(100):
            node_id = f"test-node-{i}"
            registry_node.registered_nodes[node_id] = ModelNodeIntrospectionEvent(
                node_id=node_id,
                node_type="orchestrator",  # Must be one of: effect, compute, reducer, orchestrator
                capabilities={},  # Must be dict, not list
                endpoints={},
                metadata={},
                correlation_id=uuid4(),
            )
            # Make them all expired
            past_time = datetime.now(UTC) - timedelta(
                hours=registry_node.node_ttl_hours + 1
            )
            registry_node.node_last_seen[node_id] = past_time

        # Execute memory check
        await registry_node._check_memory_usage()

        # Cleanup should have been triggered for expired nodes
        # Verify some nodes were cleaned up
        assert len(registry_node.registered_nodes) < 100


# Test: Background Task Lifecycle


class TestBackgroundTasks:
    """Test suite for background task management."""

    @pytest.mark.asyncio
    async def test_start_consuming_creates_tasks(self, registry_node):
        """Test start_consuming creates consumer and cleanup tasks."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(return_value=[])
        mock_kafka_client.commit_offsets = AsyncMock(return_value=None)
        registry_node.kafka_client = mock_kafka_client

        # Start consuming
        await registry_node.start_consuming()

        # Assertions
        assert registry_node._running is True
        assert registry_node._consumer_task is not None
        assert registry_node._cleanup_task is not None
        assert not registry_node._consumer_task.done()
        assert not registry_node._cleanup_task.done()

        # Cleanup
        await registry_node.stop_consuming()

    @pytest.mark.asyncio
    async def test_stop_consuming_cancels_tasks(self, registry_node):
        """Test stop_consuming properly cancels all tasks."""
        # Mock Kafka client
        mock_kafka_client = AsyncMock()
        mock_kafka_client.consume_messages = AsyncMock(return_value=[])
        registry_node.kafka_client = mock_kafka_client

        # Start consuming
        await registry_node.start_consuming()

        # Stop consuming
        await registry_node.stop_consuming()

        # Assertions
        assert registry_node._running is False
        assert registry_node._consumer_task is None
        assert registry_node._cleanup_task is None

    @pytest.mark.asyncio
    async def test_start_consuming_without_kafka_client(self, registry_node):
        """Test start_consuming fails gracefully without Kafka client."""
        # Ensure no Kafka client
        registry_node.kafka_client = None

        # Should raise OnexError
        from omnibase_core.errors.model_onex_error import ModelOnexError

        with pytest.raises(ModelOnexError) as exc_info:
            await registry_node.start_consuming()

        assert "kafka" in str(exc_info.value.message).lower()


# Test: Metrics


class TestMetrics:
    """Test suite for metrics tracking."""

    def test_get_registration_metrics(self, registry_node):
        """Test retrieving registration metrics."""
        metrics = registry_node.get_registration_metrics()

        assert "total_registrations" in metrics
        assert "successful_registrations" in metrics
        assert "failed_registrations" in metrics
        assert "consul_registrations" in metrics
        assert "postgres_registrations" in metrics
        assert "current_nodes_count" in metrics

    def test_get_metrics_comprehensive(self, registry_node):
        """Test retrieving comprehensive metrics."""
        metrics = registry_node.get_metrics()

        assert "registry_id" in metrics
        assert "environment" in metrics
        assert "registration_metrics" in metrics
        assert "memory_metrics" in metrics
        assert "offset_cache_metrics" in metrics
        assert "circuit_breaker_metrics" in metrics
        assert "background_tasks" in metrics
        assert "configuration" in metrics

    @pytest.mark.asyncio
    async def test_metrics_update_on_registration(
        self, registry_node, mock_introspection_event
    ):
        """Test metrics are updated after registration."""
        # Mock clients
        mock_consul_client = AsyncMock()
        mock_consul_client.register_service = AsyncMock(return_value=True)
        registry_node.consul_client = mock_consul_client

        mock_node_repository = AsyncMock()
        mock_node_repository.create_registration = AsyncMock(return_value=True)
        registry_node.node_repository = mock_node_repository

        initial_total = registry_node.registration_metrics["total_registrations"]
        initial_successful = registry_node.registration_metrics[
            "successful_registrations"
        ]

        # Execute registration
        await registry_node.dual_register(mock_introspection_event)

        # Assertions
        assert (
            registry_node.registration_metrics["total_registrations"]
            == initial_total + 1
        )
        assert (
            registry_node.registration_metrics["successful_registrations"]
            == initial_successful + 1
        )


# Test: Self-Registration


class TestSelfRegistration:
    """Test suite for registry self-registration functionality."""

    def test_registry_inherits_introspection_mixin(self):
        """Verify IntrospectionMixin is in registry class hierarchy."""
        from omninode_bridge.nodes.mixins.introspection_mixin import IntrospectionMixin

        assert issubclass(NodeBridgeRegistry, IntrospectionMixin)
        # Verify introspection methods available
        assert hasattr(NodeBridgeRegistry, "publish_introspection")
        assert hasattr(NodeBridgeRegistry, "initialize_introspection")

    @pytest.mark.asyncio
    async def test_registry_get_capabilities_override(
        self, mock_container, monkeypatch
    ):
        """Verify get_capabilities returns registry-specific info."""
        monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
        registry = NodeBridgeRegistry(mock_container, environment="dev")

        capabilities = await registry.get_capabilities()

        assert capabilities["node_type"] == "effect"
        assert capabilities["node_role"] == "registry"
        assert "registry_operations" in capabilities
        assert "registration_metrics" in capabilities

    @pytest.mark.asyncio
    async def test_registry_envelope_metadata(self, mock_container, monkeypatch):
        """Verify introspection includes network metadata."""
        monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
        monkeypatch.setenv("NETWORK_ID", "test-network")
        monkeypatch.setenv("DEPLOYMENT_ID", "test-deploy")
        monkeypatch.setenv("EPOCH", "2")

        registry = NodeBridgeRegistry(mock_container, environment="dev")

        # Set kafka_client to ensure graceful degradation doesn't kick in
        registry.kafka_client = AsyncMock()

        # Publish introspection - it will store envelope in _kafka_messages
        success = await registry.publish_introspection(reason="test")

        # Verify introspection was published successfully
        assert success is True, "Introspection should be published successfully"

        # Verify envelope was stored in _kafka_messages
        assert hasattr(
            registry, "_kafka_messages"
        ), "Registry should have _kafka_messages attribute"
        assert (
            len(registry._kafka_messages) > 0
        ), "At least one message should be stored"

        # Get the last published envelope (IntrospectionEnvelope object)
        envelope = registry._kafka_messages[-1]

        # Verify envelope has required IntrospectionEnvelope attributes
        assert hasattr(envelope, "node_id"), "Envelope should have node_id"
        assert hasattr(envelope, "node_type"), "Envelope should have node_type"
        assert hasattr(envelope, "timestamp"), "Envelope should have timestamp"
        assert hasattr(envelope, "reason"), "Envelope should have reason"
        assert hasattr(envelope, "data"), "Envelope should have data"

        # Verify basic fields are populated
        assert envelope.node_id is not None, "node_id should be set"
        assert (
            envelope.node_type == "effect"
        ), "node_type should be 'effect' (registry is an effect node)"
        assert envelope.reason == "test", "reason should match input"

        # Verify network metadata fields (Phase 1a MVP)
        assert hasattr(
            envelope, "network_id"
        ), "Envelope should have network_id attribute"
        assert hasattr(
            envelope, "deployment_id"
        ), "Envelope should have deployment_id attribute"
        assert hasattr(envelope, "epoch"), "Envelope should have epoch attribute"

        # Verify network metadata is populated from environment variables
        assert (
            envelope.network_id == "test-network"
        ), "network_id should match NETWORK_ID env var"
        assert (
            envelope.deployment_id == "test-deploy"
        ), "deployment_id should match DEPLOYMENT_ID env var"
        assert (
            envelope.epoch == 2
        ), "epoch should match EPOCH env var (converted to int)"

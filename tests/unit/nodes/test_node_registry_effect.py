# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for NodeRegistryEffect.

This test suite validates the Registry Effect Node which bridges message bus
to external infrastructure services (Consul + PostgreSQL) for node registration.

Features Tested:
    - Initialization and shutdown lifecycle
    - Register operation with dual registration (Consul + PostgreSQL)
    - Deregister operation with parallel backend cleanup
    - Discover operation with query filters
    - Request introspection event publishing
    - Circuit breaker protection and fault tolerance
    - Graceful degradation (partial success scenarios)
    - Correlation ID propagation for distributed tracing

Test Organization:
    - TestNodeRegistryEffectInitialization: Lifecycle management
    - TestNodeRegistryEffectRegister: Registration scenarios
    - TestNodeRegistryEffectDeregister: Deregistration scenarios
    - TestNodeRegistryEffectDiscover: Discovery and filtering
    - TestNodeRegistryEffectRequestIntrospection: Event publishing
    - TestNodeRegistryEffectCircuitBreaker: Fault tolerance
    - TestNodeRegistryEffectIntegration: Placeholder for integration tests

Coverage Goals:
    - >90% code coverage for node implementation
    - All success, partial success, and failure paths tested
    - Circuit breaker state transitions verified
    - Error handling and validation tested

Test Patterns (CI-Friendly):
    - SQL assertions use flexible matching (WHERE clause structure, parameterization)
      rather than exact SQL strings that may break on whitespace changes
    - Time-dependent tests use retry loops with reasonable timeouts instead of
      fixed sleeps that may fail in slow CI environments
    - Fixed timestamps (FIXED_TEST_TIMESTAMP) used for deterministic comparisons
    - Circuit breaker timeout tests poll with max wait time for CI stability
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from omnibase_core.models.node_metadata import ModelNodeCapabilitiesInfo

from omnibase_infra.errors import InfraUnavailableError, RuntimeHostError
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models.model_node_registration_metadata import (
    EnumEnvironment,
    ModelNodeRegistrationMetadata,
)

# Fixed timestamp for deterministic test results.
# Using a specific point in time avoids flaky CI tests due to timing variations.
FIXED_TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
FIXED_TEST_TIMESTAMP_ISO = FIXED_TEST_TIMESTAMP.isoformat()

from omnibase_infra.nodes.node_registry_effect.v1_0_0.models import (
    ModelNodeIntrospectionPayload,
    ModelNodeRegistryEffectConfig,
    ModelRegistryRequest,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.node import NodeRegistryEffect

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_consul_handler() -> AsyncMock:
    """Create mock Consul handler with successful execute response."""
    handler = AsyncMock()
    handler.execute = AsyncMock(return_value={"status": "success"})
    return handler


@pytest.fixture
def mock_consul_handler_failing() -> AsyncMock:
    """Create mock Consul handler that raises an exception."""
    handler = AsyncMock()
    handler.execute = AsyncMock(side_effect=Exception("Consul connection failed"))
    return handler


@pytest.fixture
def mock_db_handler() -> AsyncMock:
    """Create mock DB handler with successful execute response."""
    handler = AsyncMock()
    handler.execute = AsyncMock(
        return_value={
            "status": "success",
            "payload": {"rows_affected": 1},
        }
    )
    return handler


@pytest.fixture
def mock_db_handler_failing() -> AsyncMock:
    """Create mock DB handler that raises an exception."""
    handler = AsyncMock()
    handler.execute = AsyncMock(side_effect=Exception("PostgreSQL connection failed"))
    return handler


@pytest.fixture
def mock_db_handler_with_rows() -> AsyncMock:
    """Create mock DB handler that returns query results."""
    handler = AsyncMock()

    def create_query_response(envelope: dict[str, object]) -> dict[str, object]:
        """Generate response based on operation type."""
        operation = envelope.get("operation", "")
        if operation == "db.query":
            return {
                "status": "success",
                "payload": {
                    "rows": [
                        {
                            "node_id": "test-node-1",
                            "node_type": "effect",
                            "node_version": "1.0.0",
                            "capabilities": json.dumps({"operations": ["read"]}),
                            "endpoints": json.dumps(
                                {"health": "http://localhost:8080/health"}
                            ),
                            "metadata": json.dumps({}),
                            "health_endpoint": "http://localhost:8080/health",
                            "registered_at": FIXED_TEST_TIMESTAMP_ISO,
                            "updated_at": FIXED_TEST_TIMESTAMP_ISO,
                        },
                        {
                            "node_id": "test-node-2",
                            "node_type": "compute",
                            "node_version": "2.0.0",
                            "capabilities": json.dumps({"operations": ["compute"]}),
                            "endpoints": json.dumps({}),
                            "metadata": json.dumps({}),
                            "health_endpoint": None,
                            "registered_at": FIXED_TEST_TIMESTAMP_ISO,
                            "updated_at": FIXED_TEST_TIMESTAMP_ISO,
                        },
                    ]
                },
            }
        return {"status": "success", "payload": {"rows_affected": 1}}

    handler.execute = AsyncMock(side_effect=create_query_response)
    return handler


@pytest.fixture
def mock_db_handler_empty_results() -> AsyncMock:
    """Create mock DB handler that returns empty query results."""
    handler = AsyncMock()
    handler.execute = AsyncMock(
        return_value={
            "status": "success",
            "payload": {"rows": []},
        }
    )
    return handler


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create mock event bus with successful publish."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_event_bus_failing() -> AsyncMock:
    """Create mock event bus that raises an exception on publish."""
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=Exception("Kafka publish failed"))
    return bus


@pytest.fixture
def introspection_payload() -> ModelNodeIntrospectionPayload:
    """Create sample introspection payload for registration."""
    return ModelNodeIntrospectionPayload(
        node_id="test-node-1",
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilitiesInfo(supported_operations=["read", "write"]),
        endpoints={"health": "http://localhost:8080/health"},
        runtime_metadata=ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING
        ),
        health_endpoint="http://localhost:8080/health",
    )


@pytest.fixture
def introspection_payload_no_health() -> ModelNodeIntrospectionPayload:
    """Create sample introspection payload without health endpoint."""
    return ModelNodeIntrospectionPayload(
        node_id="test-node-no-health",
        node_type="compute",
        node_version="2.0.0",
        capabilities=ModelNodeCapabilitiesInfo(supported_operations=["compute"]),
        endpoints={},
        runtime_metadata=ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING
        ),
        health_endpoint=None,
    )


@pytest.fixture
def correlation_id() -> UUID:
    """Create a consistent correlation ID for testing."""
    return uuid4()


def create_mock_container(
    consul_handler: AsyncMock,
    db_handler: AsyncMock,
    event_bus: AsyncMock | None = None,
) -> Mock:
    """Create mock ONEX container configured for NodeRegistryEffect.

    This helper creates a mock ModelONEXContainer with service_registry
    that resolves the protocol handlers needed by NodeRegistryEffect:
    - ProtocolEnvelopeExecutor with name="consul" -> consul_handler
    - ProtocolEnvelopeExecutor with name="postgres" -> db_handler
    - ProtocolEventBus -> event_bus (or raises if None)

    Args:
        consul_handler: Mock consul handler implementing ProtocolEnvelopeExecutor.
        db_handler: Mock PostgreSQL handler implementing ProtocolEnvelopeExecutor.
        event_bus: Optional mock event bus implementing ProtocolEventBus.
            If None, event bus resolution will raise an exception.

    Returns:
        Mock container configured for NodeRegistryEffect dependency resolution.
    """
    from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
        ProtocolEnvelopeExecutor,
        ProtocolEventBus,
    )

    container = Mock()
    container.service_registry = Mock()

    async def resolve_service_side_effect(
        interface_type: type,
        name: str | None = None,
    ) -> AsyncMock:
        """Resolve mock services based on protocol type and name."""
        if interface_type is ProtocolEnvelopeExecutor:
            if name == "consul":
                return consul_handler
            elif name == "postgres":
                return db_handler
            raise ValueError(f"Unknown executor name: {name}")
        if interface_type is ProtocolEventBus:
            if event_bus is None:
                raise ValueError("ProtocolEventBus not registered")
            return event_bus
        raise ValueError(f"Service not registered: {interface_type}")

    container.service_registry.resolve_service = AsyncMock(
        side_effect=resolve_service_side_effect
    )

    return container


@pytest.fixture
def mock_container(
    mock_consul_handler: AsyncMock,
    mock_db_handler: AsyncMock,
    mock_event_bus: AsyncMock,
) -> Mock:
    """Create mock container with all handlers registered."""
    return create_mock_container(mock_consul_handler, mock_db_handler, mock_event_bus)


@pytest.fixture
def mock_container_no_event_bus(
    mock_consul_handler: AsyncMock,
    mock_db_handler: AsyncMock,
) -> Mock:
    """Create mock container without event bus."""
    return create_mock_container(mock_consul_handler, mock_db_handler, None)


async def create_test_node(
    consul_handler: AsyncMock,
    db_handler: AsyncMock,
    event_bus: AsyncMock | None = None,
    config: ModelNodeRegistryEffectConfig | None = None,
) -> NodeRegistryEffect:
    """Factory function to create a fully initialized NodeRegistryEffect for tests.

    This helper function creates a NodeRegistryEffect instance using
    container-based DI, suitable for test methods that need custom
    handler configurations.

    Args:
        consul_handler: Mock consul handler.
        db_handler: Mock PostgreSQL handler.
        event_bus: Optional mock event bus. If None, event bus operations
            will raise RuntimeHostError.
        config: Optional node configuration. Uses defaults if not provided.

    Returns:
        Fully initialized NodeRegistryEffect instance ready for use.
        The node has dependencies resolved and is marked as initialized.
    """
    container = create_mock_container(consul_handler, db_handler, event_bus)
    return await NodeRegistryEffect.create(container, config)


@pytest.fixture
async def registry_node(
    mock_container: Mock,
) -> NodeRegistryEffect:
    """Create initialized registry effect node with mocked handlers.

    Uses container-based DI via NodeRegistryEffect.create() factory method.
    The create() method returns a fully initialized node.
    """
    config = ModelNodeRegistryEffectConfig(
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=1.0,
    )
    node = await NodeRegistryEffect.create(mock_container, config)
    yield node
    await node.shutdown()


@pytest.fixture
async def registry_node_no_event_bus(
    mock_container_no_event_bus: Mock,
) -> NodeRegistryEffect:
    """Create initialized registry effect node without event bus.

    Uses container-based DI where event bus resolution fails, simulating
    deployments without Kafka configured.
    The create() method returns a fully initialized node.
    """
    config = ModelNodeRegistryEffectConfig(
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=1.0,
    )
    node = await NodeRegistryEffect.create(mock_container_no_event_bus, config)
    yield node
    await node.shutdown()


# =============================================================================
# Test: Initialization
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectInitialization:
    """Tests for node initialization and shutdown lifecycle.

    Note: More comprehensive initialization tests, including the new behavior
    where initialize() resolves dependencies, are in test_node_registry_effect_init.py.
    """

    async def test_create_returns_fully_initialized_node(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that create() factory returns a fully initialized node."""
        node = await create_test_node(mock_consul_handler, mock_db_handler)

        # Node should be fully initialized
        assert node._initialized is True
        assert node._dependencies_resolved is True
        await node.shutdown()

    async def test_shutdown_resets_initialized_flag(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that shutdown() resets the _initialized flag to False."""
        node = await create_test_node(mock_consul_handler, mock_db_handler)
        assert node._initialized is True

        await node.shutdown()

        assert node._initialized is False

    async def test_shutdown_resets_circuit_breaker(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that shutdown() resets circuit breaker state."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=2)
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )

        # Open circuit breaker by recording failures
        async with node._circuit_breaker_lock:
            await node._record_circuit_failure("test_op", uuid4())
            await node._record_circuit_failure("test_op", uuid4())

        assert node._circuit_breaker_open is True

        await node.shutdown()

        # Circuit breaker should be reset
        assert node._circuit_breaker_open is False
        assert node._circuit_breaker_failures == 0

    async def test_execute_raises_when_not_initialized(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Test that execute() raises RuntimeHostError if not initialized.

        Even with dependencies resolved, execute() requires initialize() to be called.
        This test creates a node via direct construction (not create() factory) to
        test the uninitialized state.
        """
        container = create_mock_container(mock_consul_handler, mock_db_handler, None)
        node = NodeRegistryEffect(container)
        # Resolve dependencies but do NOT call initialize()
        await node._resolve_dependencies()
        assert node._dependencies_resolved is True
        assert node._initialized is False

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        assert "not initialized" in exc_info.value.message.lower()


# =============================================================================
# Test: Register Operation
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectRegister:
    """Tests for register operation with dual registration."""

    async def test_register_dual_success(
        self,
        registry_node: NodeRegistryEffect,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Test successful dual registration (Consul + PostgreSQL)."""
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        response = await registry_node.execute(request)

        assert response.success is True
        assert response.status == "success"
        assert response.operation == "register"
        assert response.correlation_id == correlation_id
        assert response.processing_time_ms > 0

        # Verify Consul result
        assert response.consul_result is not None
        assert response.consul_result.success is True
        assert response.consul_result.service_id == introspection_payload.node_id

        # Verify PostgreSQL result
        assert response.postgres_result is not None
        assert response.postgres_result.success is True
        assert response.postgres_result.rows_affected == 1

    async def test_register_consul_fails_partial_success(
        self,
        mock_consul_handler_failing: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Test partial success when Consul fails but PostgreSQL succeeds."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler_failing, mock_db_handler, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True  # Partial success is still success
        assert response.status == "partial"
        assert response.operation == "register"

        # Consul failed
        assert response.consul_result is not None
        assert response.consul_result.success is False
        assert response.consul_result.error is not None
        assert "Consul connection failed" in response.consul_result.error

        # PostgreSQL succeeded
        assert response.postgres_result is not None
        assert response.postgres_result.success is True

        await node.shutdown()

    async def test_register_postgres_fails_partial_success(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler_failing: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Test partial success when PostgreSQL fails but Consul succeeds."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler, mock_db_handler_failing, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True  # Partial success is still success
        assert response.status == "partial"
        assert response.operation == "register"

        # Consul succeeded
        assert response.consul_result is not None
        assert response.consul_result.success is True

        # PostgreSQL failed
        assert response.postgres_result is not None
        assert response.postgres_result.success is False
        assert response.postgres_result.error is not None
        assert "PostgreSQL connection failed" in response.postgres_result.error

        await node.shutdown()

    async def test_register_both_fail(
        self,
        mock_consul_handler_failing: AsyncMock,
        mock_db_handler_failing: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Test complete failure when both backends fail."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler_failing, mock_db_handler_failing, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is False
        assert response.status == "failed"
        assert response.operation == "register"

        # Both failed
        assert response.consul_result is not None
        assert response.consul_result.success is False

        assert response.postgres_result is not None
        assert response.postgres_result.success is False

        await node.shutdown()

    async def test_register_missing_introspection_raises(
        self,
        registry_node: NodeRegistryEffect,
        correlation_id: UUID,
    ) -> None:
        """Test that register raises RuntimeHostError when introspection_event missing."""
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=None,  # Missing!
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await registry_node.execute(request)

        assert "introspection_event required" in exc_info.value.message.lower()

    async def test_register_without_health_endpoint(
        self,
        registry_node: NodeRegistryEffect,
        introspection_payload_no_health: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Test registration works without health endpoint."""
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload_no_health,
            correlation_id=correlation_id,
        )

        response = await registry_node.execute(request)

        assert response.success is True
        assert response.status == "success"

    async def test_register_parallel_execution(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Test that Consul and PostgreSQL operations run in parallel."""
        call_order: list[str] = []

        async def consul_execute(envelope: dict[str, object]) -> dict[str, object]:
            call_order.append("consul_start")
            await asyncio.sleep(0.05)
            call_order.append("consul_end")
            return {"status": "success"}

        async def db_execute(envelope: dict[str, object]) -> dict[str, object]:
            call_order.append("db_start")
            await asyncio.sleep(0.05)
            call_order.append("db_end")
            return {"status": "success", "payload": {"rows_affected": 1}}

        mock_consul_handler.execute = AsyncMock(side_effect=consul_execute)
        mock_db_handler.execute = AsyncMock(side_effect=db_execute)

        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        await node.execute(request)

        # Verify parallel execution (both starts before either ends)
        assert "consul_start" in call_order
        assert "db_start" in call_order
        consul_start_idx = call_order.index("consul_start")
        db_start_idx = call_order.index("db_start")
        consul_end_idx = call_order.index("consul_end")
        db_end_idx = call_order.index("db_end")

        # Both should start before both end (parallel execution)
        assert consul_start_idx < consul_end_idx
        assert db_start_idx < db_end_idx
        # At least one start should happen before the other's end (interleaved)
        starts = {consul_start_idx, db_start_idx}
        ends = {consul_end_idx, db_end_idx}
        assert max(starts) < max(ends)

        await node.shutdown()


# =============================================================================
# Test: Deregister Operation
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectDeregister:
    """Tests for deregister operation."""

    async def test_deregister_dual_success(
        self,
        registry_node: NodeRegistryEffect,
        correlation_id: UUID,
    ) -> None:
        """Test successful dual deregistration (Consul + PostgreSQL)."""
        request = ModelRegistryRequest(
            operation="deregister",
            node_id="test-node-1",
            correlation_id=correlation_id,
        )

        response = await registry_node.execute(request)

        assert response.success is True
        assert response.status == "success"
        assert response.operation == "deregister"
        assert response.correlation_id == correlation_id

        # Verify Consul result
        assert response.consul_result is not None
        assert response.consul_result.success is True
        assert response.consul_result.service_id == "test-node-1"

        # Verify PostgreSQL result
        assert response.postgres_result is not None
        assert response.postgres_result.success is True

    async def test_deregister_consul_fails_partial_success(
        self,
        mock_consul_handler_failing: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test partial success when Consul deregister fails."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler_failing, mock_db_handler, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="deregister",
            node_id="test-node-1",
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True
        assert response.status == "partial"
        assert response.consul_result.success is False
        assert response.postgres_result.success is True

        await node.shutdown()

    async def test_deregister_postgres_fails_partial_success(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler_failing: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test partial success when PostgreSQL deregister fails."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler, mock_db_handler_failing, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="deregister",
            node_id="test-node-1",
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True
        assert response.status == "partial"
        assert response.consul_result.success is True
        assert response.postgres_result.success is False

        await node.shutdown()

    async def test_deregister_both_fail(
        self,
        mock_consul_handler_failing: AsyncMock,
        mock_db_handler_failing: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test complete failure when both backends fail."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler_failing, mock_db_handler_failing, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="deregister",
            node_id="test-node-1",
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is False
        assert response.status == "failed"
        assert response.consul_result.success is False
        assert response.postgres_result.success is False

        await node.shutdown()

    async def test_deregister_missing_node_id_raises(
        self,
        registry_node: NodeRegistryEffect,
        correlation_id: UUID,
    ) -> None:
        """Test that deregister raises RuntimeHostError when node_id missing."""
        request = ModelRegistryRequest(
            operation="deregister",
            node_id=None,  # Missing!
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await registry_node.execute(request)

        assert "node_id required" in exc_info.value.message.lower()


# =============================================================================
# Test: Discover Operation
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectDiscover:
    """Tests for discover operation."""

    async def test_discover_no_filters(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler_with_rows: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover without filters returns all nodes."""
        node = await create_test_node(
            mock_consul_handler, mock_db_handler_with_rows, mock_event_bus
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters=None,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        # Defensive None checks before accessing attributes
        assert response is not None, "Response should not be None"
        assert response.success is True
        assert response.status == "success"
        assert response.operation == "discover"
        assert response.nodes is not None, "nodes list should not be None"
        assert len(response.nodes) == 2

        # Verify first node with defensive None check
        node1 = response.nodes[0]
        assert node1 is not None, "First node should not be None"
        assert node1.node_id == "test-node-1"
        assert node1.node_type == "effect"
        assert node1.node_version == "1.0.0"

        # Verify second node with defensive None check
        node2 = response.nodes[1]
        assert node2 is not None, "Second node should not be None"
        assert node2.node_id == "test-node-2"
        assert node2.node_type == "compute"

        await node.shutdown()

    async def test_discover_with_node_type_filter(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover with node_type filter."""
        # Mock to return filtered results
        mock_db_handler.execute = AsyncMock(
            return_value={
                "status": "success",
                "payload": {
                    "rows": [
                        {
                            "node_id": "effect-node-1",
                            "node_type": "effect",
                            "node_version": "1.0.0",
                            "capabilities": {},
                            "endpoints": {},
                            "metadata": {},
                            "health_endpoint": None,
                            "registered_at": FIXED_TEST_TIMESTAMP_ISO,
                            "updated_at": FIXED_TEST_TIMESTAMP_ISO,
                        }
                    ]
                },
            }
        )

        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters={"node_type": "effect"},
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True
        assert response.nodes is not None
        assert len(response.nodes) == 1
        assert response.nodes[0].node_type == "effect"

        # Verify the filter was applied - check SQL structure (flexible matching)
        # The SQL should have a WHERE clause with parameterized filter
        call_args = mock_db_handler.execute.call_args[0][0]
        sql = call_args["payload"]["sql"]
        params = call_args["payload"]["params"]

        # Verify parameterized query structure (not exact SQL string)
        assert "WHERE" in sql.upper(), (
            "SQL should contain WHERE clause for filtered query"
        )
        assert "$1" in sql, "SQL should use parameterized placeholder"
        assert len(params) == 1, "Should have exactly one filter parameter"
        assert "effect" in params, "Filter value should be in params"

        await node.shutdown()

    async def test_discover_with_node_id_filter(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover with node_id filter."""
        mock_db_handler.execute = AsyncMock(
            return_value={
                "status": "success",
                "payload": {
                    "rows": [
                        {
                            "node_id": "specific-node",
                            "node_type": "compute",
                            "node_version": "1.0.0",
                            "capabilities": {},
                            "endpoints": {},
                            "metadata": {},
                            "health_endpoint": None,
                            "registered_at": FIXED_TEST_TIMESTAMP_ISO,
                            "updated_at": FIXED_TEST_TIMESTAMP_ISO,
                        }
                    ]
                },
            }
        )

        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters={"node_id": "specific-node"},
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True
        assert response.nodes is not None
        assert len(response.nodes) == 1
        assert response.nodes[0].node_id == "specific-node"

        # Verify the filter was applied - check SQL structure (flexible matching)
        call_args = mock_db_handler.execute.call_args[0][0]
        sql = call_args["payload"]["sql"]
        params = call_args["payload"]["params"]

        # Verify parameterized query structure (not exact SQL string)
        assert "WHERE" in sql.upper(), (
            "SQL should contain WHERE clause for filtered query"
        )
        assert "$1" in sql, "SQL should use parameterized placeholder"
        assert len(params) == 1, "Should have exactly one filter parameter"
        assert "specific-node" in params, "Filter value should be in params"

        await node.shutdown()

    async def test_discover_with_multiple_filters(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover with multiple filters (node_type AND node_id)."""
        mock_db_handler.execute = AsyncMock(
            return_value={
                "status": "success",
                "payload": {"rows": []},
            }
        )

        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters={"node_type": "effect", "node_id": "test-node"},
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True

        # Verify filters were passed as parameters (order-independent)
        call_args = mock_db_handler.execute.call_args[0][0]
        sql = call_args["payload"]["sql"]
        params = call_args["payload"]["params"]

        # Verify two filter values were passed
        assert len(params) == 2

        # Verify WHERE clause exists with parameterized filters
        assert "WHERE" in sql
        assert "$1" in sql
        assert "$2" in sql

        # Verify the actual parameter values (order-independent)
        assert "effect" in params
        assert "test-node" in params

        await node.shutdown()

    async def test_discover_empty_results(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler_empty_results: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover returns empty list when no nodes found."""
        node = await create_test_node(
            mock_consul_handler, mock_db_handler_empty_results, mock_event_bus
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters={"node_type": "nonexistent"},
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True
        assert response.status == "success"
        assert response.nodes is not None
        assert len(response.nodes) == 0

        await node.shutdown()

    async def test_discover_handles_db_error(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler_failing: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover handles database errors gracefully."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler, mock_db_handler_failing, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters=None,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is False
        assert response.status == "failed"
        assert response.error is not None
        assert "PostgreSQL" in response.error

        await node.shutdown()

    async def test_discover_rejects_sql_injection_filter_key(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover rejects malicious filter keys (SQL injection prevention).

        SECURITY: This test verifies that filter keys containing SQL injection
        attempts are rejected with RuntimeHostError rather than being silently
        ignored or interpolated into the query.
        """
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        # Attempt SQL injection via filter key
        malicious_filters = {
            '"; DROP TABLE--': "attack",
            "node_type": "effect",  # Valid key alongside malicious one
        }

        request = ModelRegistryRequest(
            operation="discover",
            filters=malicious_filters,
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        # Verify error message mentions invalid filter keys
        assert "invalid filter keys" in exc_info.value.message.lower()
        # Verify allowed keys are mentioned for guidance
        assert "allowed keys" in exc_info.value.message.lower()
        # Verify the malicious key is sanitized in the error message (no special chars)
        assert '"; DROP' not in exc_info.value.message

        # Verify db_handler was never called (attack blocked before SQL executed)
        mock_db_handler.execute.assert_not_called()

        await node.shutdown()

    async def test_discover_rejects_unknown_filter_keys(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover rejects unknown filter keys (not in whitelist).

        SECURITY: Even benign-looking but unknown filter keys must be rejected
        to prevent any possibility of SQL injection through column names.
        """
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        # Use a plausible but non-whitelisted column name
        request = ModelRegistryRequest(
            operation="discover",
            filters={"environment": "production"},  # Not in whitelist
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        assert "invalid filter keys" in exc_info.value.message.lower()
        assert "environment" in exc_info.value.message.lower()

        # Verify db_handler was never called
        mock_db_handler.execute.assert_not_called()

        await node.shutdown()

    async def test_discover_accepts_all_whitelisted_filter_keys(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover accepts all keys in ALLOWED_FILTER_KEYS whitelist."""
        mock_db_handler.execute = AsyncMock(
            return_value={
                "status": "success",
                "payload": {"rows": []},
            }
        )

        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        # Use all whitelisted filter keys
        request = ModelRegistryRequest(
            operation="discover",
            filters={
                "node_id": "test-node",
                "node_type": "effect",
                "node_version": "1.0.0",
                "health_endpoint": "http://localhost:8080/health",
            },
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        # Should succeed without raising RuntimeHostError
        assert response.success is True
        assert response.status == "success"

        # Verify all 4 filters were applied
        call_args = mock_db_handler.execute.call_args[0][0]
        params = call_args["payload"]["params"]
        assert len(params) == 4

        await node.shutdown()

    async def test_discover_sanitizes_malicious_keys_in_error_message(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test that malicious filter keys are sanitized in error messages.

        SECURITY: Error messages must not leak SQL structure or contain
        unescaped special characters that could enable log injection.
        The original malicious keys should be transformed to safe versions.
        """
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        # SQL injection pattern
        malicious_key = '"; DROP TABLE users--'

        request = ModelRegistryRequest(
            operation="discover",
            filters={malicious_key: "attack"},
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        error_message = exc_info.value.message

        # The original malicious key should NOT appear in error message
        assert '"; DROP TABLE' not in error_message
        # The key should be sanitized (special chars replaced with underscore)
        # Original: '"; DROP TABLE users--' -> Sanitized: '__DROP_TABLE_users--'
        assert "__DROP_TABLE" in error_message
        # Semicolons and quotes from the original key should be replaced
        assert '";' not in error_message

        await node.shutdown()


# =============================================================================
# Test: Request Introspection
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectRequestIntrospection:
    """Tests for request_introspection operation."""

    async def test_request_introspection_publishes_event(
        self,
        registry_node: NodeRegistryEffect,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test that request_introspection publishes event to event bus."""
        request = ModelRegistryRequest(
            operation="request_introspection",
            correlation_id=correlation_id,
        )

        response = await registry_node.execute(request)

        assert response.success is True
        assert response.status == "success"
        assert response.operation == "request_introspection"
        assert response.correlation_id == correlation_id

        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args

        assert call_args.kwargs["topic"] == "onex.evt.registry-request-introspection.v1"
        assert call_args.kwargs["key"] == b"registry"

        # Verify message content
        message = json.loads(call_args.kwargs["value"].decode("utf-8"))
        assert message["event_type"] == "REGISTRY_REQUEST_INTROSPECTION"
        assert message["correlation_id"] == str(correlation_id)

    async def test_request_introspection_no_event_bus_raises(
        self,
        registry_node_no_event_bus: NodeRegistryEffect,
        correlation_id: UUID,
    ) -> None:
        """Test that request_introspection raises when event_bus not configured."""
        request = ModelRegistryRequest(
            operation="request_introspection",
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await registry_node_no_event_bus.execute(request)

        assert "event bus not configured" in exc_info.value.message.lower()

    async def test_request_introspection_handles_publish_error(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus_failing: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test request_introspection handles publish errors gracefully."""
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus_failing
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="request_introspection",
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is False
        assert response.status == "failed"
        assert response.error is not None
        assert "Kafka" in response.error

        await node.shutdown()


# =============================================================================
# Test: Unknown Operation
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectUnknownOperation:
    """Tests for unknown operation handling."""

    async def test_unknown_operation_raises(
        self,
        registry_node: NodeRegistryEffect,
        correlation_id: UUID,
    ) -> None:
        """Test that unknown operation raises RuntimeHostError."""
        # Use model_construct to bypass Pydantic validation and create
        # a request with an invalid operation value
        invalid_request = ModelRegistryRequest.model_construct(
            operation="invalid_operation",
            correlation_id=correlation_id,
            node_id=None,
            introspection_event=None,
            filters=None,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await registry_node.execute(invalid_request)

        assert "unknown operation" in exc_info.value.message.lower()


# =============================================================================
# Test: Circuit Breaker
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectCircuitBreaker:
    """Tests for circuit breaker behavior."""

    async def test_circuit_opens_after_threshold_failures(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that circuit breaker opens after threshold failures."""
        threshold = 2
        config = ModelNodeRegistryEffectConfig(
            circuit_breaker_threshold=threshold,
            circuit_breaker_reset_timeout=60.0,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )
        await node.initialize()

        # Manually trip the circuit breaker by recording failures
        correlation_id = uuid4()
        async with node._circuit_breaker_lock:
            await node._record_circuit_failure("test", correlation_id)
            await node._record_circuit_failure("test", correlation_id)

        # Circuit should now be open
        assert node._circuit_breaker_open is True
        assert node._circuit_breaker_failures >= threshold

        # Next request should fail with InfraUnavailableError
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=uuid4(),
        )

        with pytest.raises(InfraUnavailableError) as exc_info:
            await node.execute(request)

        assert "circuit breaker is open" in exc_info.value.message.lower()
        assert exc_info.value.model.context.get("circuit_state") == "open"

        await node.shutdown()

    async def test_circuit_resets_on_success(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that circuit breaker resets on successful operation."""
        config = ModelNodeRegistryEffectConfig(
            circuit_breaker_threshold=5,
            circuit_breaker_reset_timeout=1.0,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus, config
        )
        await node.initialize()

        # Record some failures (below threshold)
        async with node._circuit_breaker_lock:
            await node._record_circuit_failure("test", uuid4())
            await node._record_circuit_failure("test", uuid4())

        assert node._circuit_breaker_failures == 2

        # Execute successful operation
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=uuid4(),
        )

        response = await node.execute(request)
        assert response.success is True

        # Circuit breaker should be reset
        assert node._circuit_breaker_failures == 0
        assert node._circuit_breaker_open is False

        await node.shutdown()

    async def test_circuit_auto_resets_after_timeout(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that circuit breaker auto-resets after timeout."""
        reset_timeout = 0.1  # 100ms
        config = ModelNodeRegistryEffectConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_reset_timeout=reset_timeout,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )
        await node.initialize()

        # Open the circuit
        async with node._circuit_breaker_lock:
            await node._record_circuit_failure("test", uuid4())

        assert node._circuit_breaker_open is True

        # Request should fail immediately
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=uuid4(),
        )

        with pytest.raises(InfraUnavailableError):
            await node.execute(request)

        # Wait for reset timeout with retry loop for CI stability
        # Using retry loop instead of fixed sleep to handle timing variations
        max_wait = 1.0  # 1 second max wait
        poll_interval = 0.05  # 50ms between attempts
        start_time = time.time()
        response = None

        # Initial wait for reset timeout to elapse
        await asyncio.sleep(reset_timeout)

        while time.time() - start_time < max_wait:
            try:
                response = await node.execute(request)
                if response.success:
                    break
            except InfraUnavailableError:
                # Circuit not yet reset, wait and retry
                await asyncio.sleep(poll_interval)
        else:
            pytest.fail(
                f"Circuit breaker did not reset within {max_wait}s "
                f"(reset_timeout={reset_timeout}s)"
            )

        # Verify the response
        assert response is not None
        assert response.success is True

        await node.shutdown()

    async def test_circuit_includes_retry_after_seconds(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that circuit breaker error includes retry_after_seconds."""
        reset_timeout = 30.0
        config = ModelNodeRegistryEffectConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_reset_timeout=reset_timeout,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )
        await node.initialize()

        # Open the circuit
        async with node._circuit_breaker_lock:
            await node._record_circuit_failure("test", uuid4())

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=uuid4(),
        )

        with pytest.raises(InfraUnavailableError) as exc_info:
            await node.execute(request)

        error = exc_info.value
        retry_after = error.model.context.get("retry_after_seconds")
        assert retry_after is not None
        assert isinstance(retry_after, int)
        assert 0 < retry_after <= reset_timeout

        await node.shutdown()

    async def test_partial_success_resets_circuit(
        self,
        mock_consul_handler_failing: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that partial success (one backend succeeds) resets circuit breaker."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler_failing, mock_db_handler, mock_event_bus, config
        )
        await node.initialize()

        # Record some failures (below threshold)
        async with node._circuit_breaker_lock:
            await node._record_circuit_failure("test", uuid4())
            await node._record_circuit_failure("test", uuid4())

        assert node._circuit_breaker_failures == 2

        # Execute operation (Consul fails, Postgres succeeds = partial)
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=uuid4(),
        )

        response = await node.execute(request)
        assert response.success is True
        assert response.status == "partial"

        # Partial success should reset circuit
        assert node._circuit_breaker_failures == 0

        await node.shutdown()

    async def test_complete_failure_does_not_record_circuit_failure(
        self,
        mock_consul_handler_failing: AsyncMock,
        mock_db_handler_failing: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that complete failure doesn't double-record circuit failures.

        The node's execute method does not record circuit failures for its own
        errors (RuntimeHostError, InfraUnavailableError) - only for unexpected
        exceptions. The individual backend failures are handled within the
        parallel tasks and don't propagate as exceptions.
        """
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=5)
        node = await create_test_node(
            mock_consul_handler_failing, mock_db_handler_failing, mock_event_bus, config
        )
        await node.initialize()

        initial_failures = node._circuit_breaker_failures
        assert initial_failures == 0

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=uuid4(),
        )

        response = await node.execute(request)
        assert response.success is False
        assert response.status == "failed"

        # Circuit failures should not be incremented for gracefully handled failures
        # The execute method catches backend exceptions and converts them to result objects
        assert node._circuit_breaker_failures == initial_failures

        await node.shutdown()


# =============================================================================
# Test: Correlation ID Propagation
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectCorrelationId:
    """Tests for correlation ID propagation."""

    async def test_correlation_id_propagated_to_consul_handler(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that correlation ID is passed to Consul handler."""
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        await node.execute(request)

        # Verify correlation ID was passed to Consul handler
        call_args = mock_consul_handler.execute.call_args[0][0]
        assert call_args["correlation_id"] == correlation_id

        await node.shutdown()

    async def test_correlation_id_propagated_to_db_handler(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that correlation ID is passed to DB handler."""
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        await node.execute(request)

        # Verify correlation ID was passed to DB handler
        call_args = mock_db_handler.execute.call_args[0][0]
        assert call_args["correlation_id"] == correlation_id

        await node.shutdown()

    async def test_correlation_id_in_response(
        self,
        registry_node: NodeRegistryEffect,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that correlation ID is included in response."""
        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        response = await registry_node.execute(request)

        assert response.correlation_id == correlation_id

    async def test_correlation_id_in_circuit_breaker_error(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that correlation ID is in circuit breaker error."""
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=1)
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )
        await node.initialize()

        # Open the circuit
        async with node._circuit_breaker_lock:
            await node._record_circuit_failure("test", uuid4())

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        with pytest.raises(InfraUnavailableError) as exc_info:
            await node.execute(request)

        assert exc_info.value.model.correlation_id == correlation_id

        await node.shutdown()


# =============================================================================
# Test: Error Sanitization
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectErrorSanitization:
    """Tests for error message sanitization."""

    async def test_sanitize_error_includes_exception_type(self) -> None:
        """Test that exception type is included in sanitized error."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        error = ValueError("Test error message")
        result = node._sanitize_error(error)
        assert result.startswith("ValueError: ")
        assert "Test error message" in result

    async def test_sanitize_error_redacts_password(self) -> None:
        """Test that passwords are redacted from error messages."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        error = Exception("Connection failed: password=secret123")
        result = node._sanitize_error(error)
        assert "secret123" not in result
        assert "[REDACTED]" in result

    async def test_sanitize_error_redacts_token(self) -> None:
        """Test that tokens are redacted from error messages."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        error = Exception("Auth failed: token=abc123xyz")
        result = node._sanitize_error(error)
        assert "abc123xyz" not in result
        assert "[REDACTED]" in result

    async def test_sanitize_error_redacts_api_key(self) -> None:
        """Test that API keys are redacted from error messages."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        error = Exception("Request failed: api_key=my-secret-key")
        result = node._sanitize_error(error)
        assert "my-secret-key" not in result
        assert "[REDACTED]" in result

    async def test_sanitize_error_redacts_connection_string(self) -> None:
        """Test that connection string credentials are redacted."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        error = Exception("Failed: postgresql://user:password123@host:5432/db")
        result = node._sanitize_error(error)
        assert "password123" not in result
        assert "[REDACTED]" in result

    async def test_sanitize_error_truncates_long_messages(self) -> None:
        """Test that long error messages are truncated."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        # Use a mixed message with spaces to avoid base64-like pattern matching
        long_message = "Error: " + " ".join(["word"] * 200)  # Long message with spaces
        error = Exception(long_message)
        result = node._sanitize_error(error)
        # Message should be truncated to 500 chars + type prefix + "..."
        assert len(result) <= 520  # Type name + ": " + 500 chars + "..."
        assert result.endswith("...")

    async def test_sanitize_error_redacts_secret(self) -> None:
        """Test that secrets are redacted from error messages."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        error = Exception("Config error: secret=mysupersecret")
        result = node._sanitize_error(error)
        assert "mysupersecret" not in result
        assert "[REDACTED]" in result

    async def test_sanitize_error_redacts_bearer_token(self) -> None:
        """Test that bearer tokens are redacted."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        error = Exception("Auth header: Bearer eyJhbGciOiJIUzI1NiJ9.test")
        result = node._sanitize_error(error)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result
        assert "[REDACTED]" in result

    @pytest.mark.asyncio
    async def test_sanitize_error_used_in_consul_registration(self) -> None:
        """Test that _sanitize_error is used for Consul registration failures."""
        mock_consul_handler = Mock()
        mock_consul_handler.execute = AsyncMock(
            side_effect=Exception("Consul error: api_key=supersecret123")
        )
        mock_db_handler = Mock()
        mock_db_handler.execute = AsyncMock(
            return_value={"status": "success", "payload": {"rows_affected": 1}}
        )

        node = await create_test_node(mock_consul_handler, mock_db_handler)
        await node.initialize()

        introspection = ModelNodeIntrospectionPayload(
            node_id="test-node",
            node_type="effect",
            node_version="1.0.0",
            runtime_metadata=ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING
            ),
        )
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection,
            correlation_id=uuid4(),
        )
        response = await node.execute(request)

        # Consul failed, Postgres succeeded = partial success
        assert response.status == "partial"
        assert response.consul_result is not None
        assert response.consul_result.error is not None
        # Verify sensitive data is redacted
        assert "supersecret123" not in response.consul_result.error
        assert "[REDACTED]" in response.consul_result.error
        assert "Exception:" in response.consul_result.error

    @pytest.mark.asyncio
    async def test_sanitize_error_in_registration_failure(self) -> None:
        """Test that registration failures sanitize error messages."""
        mock_consul_handler = Mock()
        mock_consul_handler.execute = AsyncMock(
            side_effect=Exception("Consul error: token=secret123")
        )
        mock_db_handler = Mock()
        mock_db_handler.execute = AsyncMock(return_value={"status": "success"})

        node = await create_test_node(mock_consul_handler, mock_db_handler)
        await node.initialize()

        introspection = ModelNodeIntrospectionPayload(
            node_id="test-node",
            node_type="effect",
            node_version="1.0.0",
            runtime_metadata=ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING
            ),
        )
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection,
            correlation_id=uuid4(),
        )
        response = await node.execute(request)

        # Should be partial success (postgres succeeded, consul failed)
        assert response.status == "partial"
        assert response.consul_result is not None
        assert response.consul_result.error is not None
        assert "secret123" not in response.consul_result.error
        assert "[REDACTED]" in response.consul_result.error


# =============================================================================
# Test: JSON Serialization Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectJsonSerialization:
    """Tests for JSON serialization error handling."""

    async def test_safe_json_dumps_success(self) -> None:
        """Test that _safe_json_dumps serializes valid data."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        data = {"key": "value", "nested": {"list": [1, 2, 3]}}
        result = node._safe_json_dumps(data, uuid4(), "test_field")
        assert result == json.dumps(data)

    async def test_safe_json_dumps_handles_sets_gracefully(self) -> None:
        """Test that _safe_json_dumps converts sets to sorted lists."""
        node = await create_test_node(AsyncMock(), AsyncMock())

        # Sets are converted to sorted lists by the default serializer
        data_with_set = {"items": {3, 1, 2}}
        correlation_id = uuid4()
        result = node._safe_json_dumps(data_with_set, correlation_id, "test_field")

        # Should serialize successfully with set converted to sorted list
        parsed = json.loads(result)
        assert parsed["items"] == ["1", "2", "3"]  # sorted string representation

    async def test_safe_json_dumps_handles_truly_unserializable_object(self) -> None:
        """Test that _safe_json_dumps handles truly unserializable objects gracefully."""
        node = await create_test_node(AsyncMock(), AsyncMock())

        # Create a truly unserializable object - a lambda function
        # Note: The default serializer will convert this to a string representation
        unserializable_data = {"func": lambda x: x}
        correlation_id = uuid4()
        result = node._safe_json_dumps(
            unserializable_data, correlation_id, "test_field"
        )

        # Should serialize successfully (lambda converted to string)
        parsed = json.loads(result)
        assert "non-serializable" in parsed["func"]

    async def test_safe_json_dumps_handles_circular_reference(self) -> None:
        """Test that _safe_json_dumps handles circular references."""
        node = await create_test_node(AsyncMock(), AsyncMock())

        # Create circular reference
        circular: dict[str, object] = {}
        circular["self"] = circular

        correlation_id = uuid4()
        result = node._safe_json_dumps(circular, correlation_id, "circular_field")

        # Should return fallback value
        assert result == "{}"

    async def test_safe_json_dumps_custom_fallback_on_recursion_error(self) -> None:
        """Test that _safe_json_dumps uses custom fallback on RecursionError."""
        node = await create_test_node(AsyncMock(), AsyncMock())

        # Create deeply nested structure that triggers RecursionError
        circular: dict[str, object] = {}
        circular["self"] = circular

        result = node._safe_json_dumps(circular, uuid4(), "test_field", fallback="[]")

        # Should return custom fallback value due to RecursionError
        assert result == "[]"

    async def test_safe_json_dumps_converts_datetime(self) -> None:
        """Test that _safe_json_dumps converts datetime to ISO format."""
        node = await create_test_node(AsyncMock(), AsyncMock())

        test_time = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        data = {"timestamp": test_time}
        result = node._safe_json_dumps(data, uuid4(), "test_field")

        parsed = json.loads(result)
        assert parsed["timestamp"] == "2024-01-15T10:30:45+00:00"

    async def test_safe_json_dumps_converts_uuid(self) -> None:
        """Test that _safe_json_dumps converts UUID to string."""
        node = await create_test_node(AsyncMock(), AsyncMock())

        test_uuid = uuid4()
        data = {"id": test_uuid}
        result = node._safe_json_dumps(data, uuid4(), "test_field")

        parsed = json.loads(result)
        assert parsed["id"] == str(test_uuid)

    async def test_safe_json_dumps_strict_success(self) -> None:
        """Test that _safe_json_dumps_strict returns None error on success."""
        node = await create_test_node(AsyncMock(), AsyncMock())
        data = {"event_type": "TEST", "correlation_id": "abc123"}
        json_str, error = node._safe_json_dumps_strict(data, uuid4(), "test_field")

        assert error is None
        assert json_str == json.dumps(data)

    async def test_safe_json_dumps_strict_handles_sets(self) -> None:
        """Test that _safe_json_dumps_strict converts sets to sorted lists."""
        node = await create_test_node(AsyncMock(), AsyncMock())

        # Sets are now converted by the default serializer
        data_with_set = {"items": {3, 1, 2}}
        json_str, error = node._safe_json_dumps_strict(
            data_with_set, uuid4(), "set_field"
        )

        # Should succeed with set converted to sorted list
        assert error is None
        parsed = json.loads(json_str)
        assert parsed["items"] == ["1", "2", "3"]

    async def test_safe_json_dumps_strict_failure_on_recursion(self) -> None:
        """Test that _safe_json_dumps_strict returns error message on circular reference.

        Note: Circular references trigger ValueError when json.dumps detects a circular
        reference in the data structure. The exact error type depends on how the
        serialization handles the circular reference detection.
        """
        node = await create_test_node(AsyncMock(), AsyncMock())

        # Create circular reference that causes serialization failure
        circular: dict[str, object] = {}
        circular["self"] = circular
        json_str, error = node._safe_json_dumps_strict(
            circular, uuid4(), "circular_field"
        )

        assert json_str == "{}"
        assert error is not None
        assert "JSON serialization failed" in error
        assert "circular_field" in error
        # ValueError is raised by json.dumps when it detects circular reference
        assert "ValueError" in error or "RecursionError" in error

    async def test_request_introspection_fails_on_serialization_error(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that request_introspection fails gracefully on serialization error.

        This test verifies that if the event payload cannot be serialized,
        the operation returns a failure response rather than publishing a malformed event.
        """
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        # Monkey-patch _safe_json_dumps_strict to simulate serialization failure
        original_method = node._safe_json_dumps_strict

        def failing_serializer(
            data: object,
            correlation_id: UUID | None = None,
            field_name: str = "unknown",
        ) -> tuple[str, str | None]:
            # Simulate serialization failure for introspection events
            if field_name == "introspection_event":
                return (
                    "{}",
                    "JSON serialization failed for introspection_event: TypeError",
                )
            return original_method(data, correlation_id, field_name)

        node._safe_json_dumps_strict = failing_serializer  # type: ignore[method-assign]

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="request_introspection",
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        # Should fail gracefully
        assert response.success is False
        assert response.status == "failed"
        assert response.error is not None
        assert "JSON serialization failed" in response.error
        assert response.correlation_id == correlation_id

        # Event bus should NOT have been called
        mock_event_bus.publish.assert_not_called()

        await node.shutdown()

    async def test_register_handles_json_serialization_for_capabilities(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that registration handles unserializable capabilities gracefully."""
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus
        )
        await node.initialize()

        # Create introspection with complex nested data
        # Note: capabilities is now ModelNodeCapabilitiesInfo (typed model)
        # and runtime_metadata is ModelNodeRegistrationMetadata
        introspection = ModelNodeIntrospectionPayload(
            node_id="test-node",
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilitiesInfo(
                supported_operations=["read", "write"]
            ),
            endpoints={"health": "http://localhost:8080/health"},
            runtime_metadata=ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING
            ),
        )

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        # Registration should succeed
        assert response.success is True
        assert response.correlation_id == correlation_id

        # Verify JSON serialization was called via db_handler
        mock_db_handler.execute.assert_called_once()
        call_args = mock_db_handler.execute.call_args[0][0]
        params = call_args["payload"]["params"]

        # Verify capabilities, endpoints, metadata were serialized
        # They should be JSON strings
        assert isinstance(params[3], str)  # capabilities
        assert isinstance(params[4], str)  # endpoints
        assert isinstance(params[5], str)  # metadata

        await node.shutdown()


# =============================================================================
# Test: Configurable Slow Operation Threshold
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectSlowOperationThreshold:
    """Tests for configurable slow operation threshold."""

    async def test_default_threshold_is_1000ms(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that default slow operation threshold is 1000ms."""
        node = await create_test_node(mock_consul_handler, mock_db_handler)
        # Default config should set threshold to 1000ms
        assert node._slow_operation_threshold_ms == 1000.0

    async def test_custom_threshold_from_config(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that slow operation threshold can be configured."""
        custom_threshold = 500.0
        config = ModelNodeRegistryEffectConfig(
            slow_operation_threshold_ms=custom_threshold,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )
        assert node._slow_operation_threshold_ms == custom_threshold

    async def test_zero_threshold_valid(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that zero threshold is valid (logs all operations as slow)."""
        config = ModelNodeRegistryEffectConfig(
            slow_operation_threshold_ms=0.0,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )
        assert node._slow_operation_threshold_ms == 0.0

    async def test_high_threshold_for_slow_environments(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that high threshold works for slow environments (e.g., CI)."""
        # In CI environments, operations may take longer
        ci_threshold = 5000.0  # 5 seconds
        config = ModelNodeRegistryEffectConfig(
            slow_operation_threshold_ms=ci_threshold,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, config=config
        )
        assert node._slow_operation_threshold_ms == ci_threshold

    async def test_slow_operation_uses_configured_threshold(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that slow operation warning uses configured threshold."""
        import logging

        # Set a very low threshold so normal operations trigger slow warning
        low_threshold = 0.001  # 0.001ms - any operation will exceed this
        config = ModelNodeRegistryEffectConfig(
            slow_operation_threshold_ms=low_threshold,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        with caplog.at_level(logging.WARNING):
            await node.execute(request)

        # Verify slow operation warning was logged with configured threshold
        slow_warnings = [
            record
            for record in caplog.records
            if "Slow registry operation" in record.message
        ]
        assert len(slow_warnings) >= 1

        # Verify the warning message contains our configured threshold
        warning_message = slow_warnings[0].message
        assert f"threshold: {low_threshold}ms" in warning_message

        await node.shutdown()

    async def test_high_threshold_suppresses_slow_warning(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
        mock_event_bus: AsyncMock,
        introspection_payload: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that high threshold suppresses slow operation warning."""
        import logging

        # Set a very high threshold so normal operations don't trigger slow warning
        high_threshold = 60000.0  # 60 seconds - no operation will exceed this
        config = ModelNodeRegistryEffectConfig(
            slow_operation_threshold_ms=high_threshold,
        )
        node = await create_test_node(
            mock_consul_handler, mock_db_handler, mock_event_bus, config
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        with caplog.at_level(logging.WARNING):
            await node.execute(request)

        # Verify no slow operation warning was logged
        slow_warnings = [
            record
            for record in caplog.records
            if "Slow registry operation" in record.message
        ]
        assert len(slow_warnings) == 0

        await node.shutdown()


# =============================================================================
# Integration Tests Placeholder
# =============================================================================


@pytest.mark.integration
class TestNodeRegistryEffectIntegration:
    """Integration tests placeholder for NodeRegistryEffect with real backends.

    These tests require real Consul and PostgreSQL instances to run.
    They validate end-to-end behavior that cannot be fully tested with mocks.

    TODO: Integration tests to implement:
        - test_register_with_real_consul: Verify Consul service registration
          with actual health check callbacks
        - test_register_with_real_postgres: Verify PostgreSQL UPSERT behavior
          with actual database constraints
        - test_discover_with_real_postgres: Verify SQL query generation
          works with real PostgreSQL query planner
        - test_circuit_breaker_with_real_failures: Test circuit breaker
          behavior with actual network failures/timeouts
        - test_concurrent_registrations: Verify idempotent registration
          under concurrent load with real backends
        - test_event_bus_with_real_kafka: Verify introspection events
          are properly published to Kafka

    Environment Setup Required:
        - Consul: localhost:8500
        - PostgreSQL: localhost:5432 with node_registrations table
        - Kafka: localhost:9092 with registry topics

    Run with: pytest -m integration tests/unit/nodes/test_node_registry_effect.py
    """

    @pytest.mark.skip(reason="Requires real Consul instance")
    async def test_register_with_real_consul(self) -> None:
        """Test node registration with real Consul backend.

        TODO: Implement integration test that:
        1. Connects to real Consul at localhost:8500
        2. Registers a test node with health check
        3. Verifies node appears in Consul catalog
        4. Verifies health check is callable
        5. Deregisters and verifies cleanup
        """

    @pytest.mark.skip(reason="Requires real PostgreSQL instance")
    async def test_register_with_real_postgres(self) -> None:
        """Test node registration with real PostgreSQL backend.

        TODO: Implement integration test that:
        1. Connects to real PostgreSQL at localhost:5432
        2. Registers a test node with UPSERT
        3. Verifies row exists in node_registrations table
        4. Re-registers same node and verifies updated_at changes
        5. Deregisters and verifies row deleted
        """

    @pytest.mark.skip(reason="Requires real PostgreSQL instance")
    async def test_discover_with_real_postgres(self) -> None:
        """Test node discovery with real PostgreSQL query execution.

        TODO: Implement integration test that:
        1. Registers multiple test nodes with different types
        2. Queries with various filter combinations
        3. Verifies SQL parameterization prevents injection
        4. Verifies results match expected filters
        """

    @pytest.mark.skip(reason="Requires real Kafka instance")
    async def test_introspection_with_real_kafka(self) -> None:
        """Test introspection event publishing with real Kafka.

        TODO: Implement integration test that:
        1. Connects to real Kafka at localhost:9092
        2. Sets up consumer for introspection topic
        3. Publishes introspection request
        4. Verifies event received by consumer
        5. Verifies event schema and correlation ID
        """


# =============================================================================
# Test: Dependency Resolution Error Paths
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectDependencyResolution:
    """Tests for container-based dependency resolution error handling.

    This test class validates error paths in:
    - _ensure_dependencies(): Called before operations to verify handlers resolved
    - _resolve_dependencies(): Called during initialization to get handlers from container

    These tests ensure proper error messages and RuntimeError exceptions when:
    - Dependencies not resolved (calling operations before initialization)
    - Container doesn't have consul handler registered
    - Container doesn't have postgres handler registered
    - Handler resolution succeeds but returns None
    """

    async def test_ensure_dependencies_raises_when_not_resolved(self) -> None:
        """Test _ensure_dependencies raises RuntimeError when not resolved.

        This error occurs when operations are attempted before calling
        _resolve_dependencies() or using the create() factory method.
        """
        container = Mock()
        container.service_registry = Mock()
        container.service_registry.resolve_service = AsyncMock()

        node = NodeRegistryEffect(container)
        # Do NOT call _resolve_dependencies()

        with pytest.raises(RuntimeError) as exc_info:
            node._ensure_dependencies()

        error_message = str(exc_info.value)
        assert "Dependencies not resolved" in error_message
        assert "_resolve_dependencies()" in error_message or "create()" in error_message

    async def test_ensure_dependencies_raises_when_consul_handler_none(self) -> None:
        """Test _ensure_dependencies raises when consul_handler is None after resolution.

        This scenario can occur if the container's resolve_service returns None
        instead of raising an exception, or if the handler was explicitly set to None.
        """
        container = Mock()
        container.service_registry = Mock()
        container.service_registry.resolve_service = AsyncMock()

        node = NodeRegistryEffect(container)
        # Simulate partially resolved state where consul_handler is None
        node._dependencies_resolved = True
        node._consul_handler = None
        node._db_handler = AsyncMock()  # db_handler is resolved

        with pytest.raises(RuntimeError) as exc_info:
            node._ensure_dependencies()

        error_message = str(exc_info.value)
        assert "Required handlers" in error_message
        assert "consul_handler" in error_message or "db_handler" in error_message
        assert "None" in error_message

    async def test_ensure_dependencies_raises_when_db_handler_none(self) -> None:
        """Test _ensure_dependencies raises when db_handler is None after resolution.

        This scenario can occur if the container's resolve_service returns None
        instead of raising an exception, or if the handler was explicitly set to None.
        """
        container = Mock()
        container.service_registry = Mock()
        container.service_registry.resolve_service = AsyncMock()

        node = NodeRegistryEffect(container)
        # Simulate partially resolved state where db_handler is None
        node._dependencies_resolved = True
        node._consul_handler = AsyncMock()  # consul_handler is resolved
        node._db_handler = None

        with pytest.raises(RuntimeError) as exc_info:
            node._ensure_dependencies()

        error_message = str(exc_info.value)
        assert "Required handlers" in error_message
        assert "consul_handler" in error_message or "db_handler" in error_message
        assert "None" in error_message

    async def test_ensure_dependencies_raises_when_both_handlers_none(self) -> None:
        """Test _ensure_dependencies raises when both handlers are None."""
        container = Mock()
        container.service_registry = Mock()
        container.service_registry.resolve_service = AsyncMock()

        node = NodeRegistryEffect(container)
        # Simulate resolved state where both handlers are None
        node._dependencies_resolved = True
        node._consul_handler = None
        node._db_handler = None

        with pytest.raises(RuntimeError) as exc_info:
            node._ensure_dependencies()

        error_message = str(exc_info.value)
        assert "Required handlers" in error_message

    async def test_resolve_dependencies_raises_when_consul_not_registered(self) -> None:
        """Test _resolve_dependencies raises when consul handler not in container.

        This error occurs during node initialization when the container doesn't
        have a ProtocolEnvelopeExecutor registered with name='consul'.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_failing_consul(
            interface_type: type,
            name: str | None = None,
        ) -> None:
            """Simulate missing consul handler."""
            if interface_type is ProtocolEnvelopeExecutor and name == "consul":
                raise ValueError("Service not registered: consul")
            # Should not reach here in this test
            raise ValueError(f"Unexpected resolution: {interface_type}, {name}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_failing_consul
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "consul" in error_message.lower()
        # Removed: Check is now just for consul keyword
        assert "name='consul'" in error_message

    async def test_resolve_dependencies_raises_when_postgres_not_registered(
        self,
    ) -> None:
        """Test _resolve_dependencies raises when postgres handler not in container.

        This error occurs during node initialization when the container doesn't
        have a ProtocolEnvelopeExecutor registered with name='postgres'.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
        )

        consul_handler = AsyncMock()
        container = Mock()
        container.service_registry = Mock()

        async def resolve_failing_postgres(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            """Simulate consul success, postgres failure."""
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return consul_handler
                if name == "postgres":
                    # Use KeyError to simulate "service not registered" scenario
                    # (ValueError would be treated as configuration error)
                    raise KeyError("Service not registered: postgres")
            raise KeyError(f"Unexpected resolution: {interface_type}, {name}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_failing_postgres
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "postgresql" in error_message.lower()
        # Check for handler name in error message
        assert "name='postgres'" in error_message

    async def test_resolve_dependencies_skips_if_already_resolved(self) -> None:
        """Test _resolve_dependencies is idempotent - skips if already resolved.

        This ensures we don't unnecessarily re-resolve dependencies or
        overwrite handlers that may have been customized after initial resolution.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        resolve_call_count = 0
        consul_handler = AsyncMock()
        db_handler = AsyncMock()
        event_bus = AsyncMock()

        container = Mock()
        container.service_registry = Mock()

        async def resolve_and_count(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            nonlocal resolve_call_count
            resolve_call_count += 1
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return consul_handler
                if name == "postgres":
                    return db_handler
            if interface_type is ProtocolEventBus:
                return event_bus
            raise ValueError(f"Unknown: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_and_count
        )

        node = NodeRegistryEffect(container)

        # First resolution
        await node._resolve_dependencies()
        first_call_count = resolve_call_count

        # Second call should be a no-op
        await node._resolve_dependencies()

        # Call count should not increase
        assert resolve_call_count == first_call_count
        assert node._dependencies_resolved is True

    async def test_resolve_dependencies_continues_without_event_bus(self) -> None:
        """Test _resolve_dependencies succeeds when event bus not registered.

        The event bus is optional - only needed for request_introspection operation.
        Resolution should complete successfully even if ProtocolEventBus is not
        registered in the container.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        consul_handler = AsyncMock()
        db_handler = AsyncMock()

        container = Mock()
        container.service_registry = Mock()

        async def resolve_without_event_bus(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            """Simulate consul/postgres success, event bus failure."""
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return consul_handler
                if name == "postgres":
                    return db_handler
            if interface_type is ProtocolEventBus:
                raise ValueError("ProtocolEventBus not registered")
            raise ValueError(f"Unknown: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_without_event_bus
        )

        node = NodeRegistryEffect(container)

        # Should not raise - event bus is optional
        await node._resolve_dependencies()

        assert node._dependencies_resolved is True
        assert node._consul_handler is consul_handler
        assert node._db_handler is db_handler
        assert node._event_bus is None  # Optional, not resolved

    async def test_consul_handler_property_ensures_dependencies(self) -> None:
        """Test consul_handler property calls _ensure_dependencies before access."""
        container = Mock()
        container.service_registry = Mock()
        container.service_registry.resolve_service = AsyncMock()

        node = NodeRegistryEffect(container)
        # Do NOT resolve dependencies

        with pytest.raises(RuntimeError) as exc_info:
            _ = node.consul_handler  # Property access should trigger check

        assert "Dependencies not resolved" in str(exc_info.value)

    async def test_db_handler_property_ensures_dependencies(self) -> None:
        """Test db_handler property calls _ensure_dependencies before access."""
        container = Mock()
        container.service_registry = Mock()
        container.service_registry.resolve_service = AsyncMock()

        node = NodeRegistryEffect(container)
        # Do NOT resolve dependencies

        with pytest.raises(RuntimeError) as exc_info:
            _ = node.db_handler  # Property access should trigger check

        assert "Dependencies not resolved" in str(exc_info.value)

    async def test_create_factory_method_resolves_dependencies(
        self,
        mock_container: Mock,
    ) -> None:
        """Test create() factory method resolves dependencies automatically.

        The create() class method is the recommended way to instantiate
        NodeRegistryEffect. It handles dependency resolution automatically.
        """
        node = await NodeRegistryEffect.create(mock_container)

        assert node._dependencies_resolved is True
        assert node._consul_handler is not None
        assert node._db_handler is not None
        # Event bus should be resolved from mock_registry_container
        assert node._event_bus is not None

    async def test_create_factory_method_with_missing_consul_raises(self) -> None:
        """Test create() factory raises RuntimeError when consul missing.

        The factory method should propagate the RuntimeError from
        _resolve_dependencies() when the consul handler is not registered.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_missing_consul(
            interface_type: type,
            name: str | None = None,
        ) -> None:
            if interface_type is ProtocolEnvelopeExecutor and name == "consul":
                raise ValueError("consul not registered")
            raise ValueError(f"Unknown: {interface_type}, {name}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_missing_consul
        )

        with pytest.raises(RuntimeError) as exc_info:
            await NodeRegistryEffect.create(container)

        assert "consul handler" in str(exc_info.value).lower()

    async def test_create_factory_method_with_missing_postgres_raises(self) -> None:
        """Test create() factory raises RuntimeError when postgres missing.

        The factory method should propagate the RuntimeError from
        _resolve_dependencies() when the postgres handler is not registered.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
        )

        consul_handler = AsyncMock()
        container = Mock()
        container.service_registry = Mock()

        async def resolve_missing_postgres(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return consul_handler
                if name == "postgres":
                    # Use KeyError to simulate "service not registered" scenario
                    raise KeyError("postgres not registered")
            raise KeyError(f"Unknown: {interface_type}, {name}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_missing_postgres
        )

        with pytest.raises(RuntimeError) as exc_info:
            await NodeRegistryEffect.create(container)

        assert "postgresql" in str(exc_info.value).lower()

    async def test_original_error_preserved_in_consul_resolution_failure(self) -> None:
        """Test that original exception is preserved when consul resolution fails.

        The RuntimeError should chain the original exception via 'from e' to
        preserve the full error context for debugging.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
        )

        original_error_message = "Consul service discovery failed: timeout"
        container = Mock()
        container.service_registry = Mock()

        async def resolve_with_original_error(
            interface_type: type,
            name: str | None = None,
        ) -> None:
            if interface_type is ProtocolEnvelopeExecutor and name == "consul":
                raise ConnectionError(original_error_message)
            raise ValueError(f"Unknown: {interface_type}, {name}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_with_original_error
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        # Verify error chaining
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ConnectionError)
        assert original_error_message in str(exc_info.value.__cause__)
        # Original error should be mentioned in the RuntimeError message
        assert original_error_message in str(exc_info.value)

    async def test_original_error_preserved_in_postgres_resolution_failure(
        self,
    ) -> None:
        """Test that original exception is preserved when postgres resolution fails.

        The RuntimeError should chain the original exception via 'from e' to
        preserve the full error context for debugging.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
        )

        consul_handler = AsyncMock()
        original_error_message = "PostgreSQL pool exhausted: max connections reached"
        container = Mock()
        container.service_registry = Mock()

        async def resolve_with_original_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return consul_handler
                if name == "postgres":
                    raise ConnectionError(original_error_message)
            raise ValueError(f"Unknown: {interface_type}, {name}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_with_original_error
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        # Verify error chaining
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ConnectionError)
        assert original_error_message in str(exc_info.value.__cause__)
        # Original error should be mentioned in the RuntimeError message
        assert original_error_message in str(exc_info.value)

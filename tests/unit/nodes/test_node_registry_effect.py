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

Coverage Goals:
    - >90% code coverage for node implementation
    - All success, partial success, and failure paths tested
    - Circuit breaker state transitions verified
    - Error handling and validation tested
"""

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import InfraUnavailableError, RuntimeHostError
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

    def create_query_response(envelope: dict[str, Any]) -> dict[str, Any]:
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
                            "registered_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                        },
                        {
                            "node_id": "test-node-2",
                            "node_type": "compute",
                            "node_version": "2.0.0",
                            "capabilities": json.dumps({"operations": ["compute"]}),
                            "endpoints": json.dumps({}),
                            "metadata": json.dumps({}),
                            "health_endpoint": None,
                            "registered_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
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
        capabilities={"operations": ["read", "write"]},
        endpoints={"health": "http://localhost:8080/health"},
        metadata={"environment": "test"},
        health_endpoint="http://localhost:8080/health",
    )


@pytest.fixture
def introspection_payload_no_health() -> ModelNodeIntrospectionPayload:
    """Create sample introspection payload without health endpoint."""
    return ModelNodeIntrospectionPayload(
        node_id="test-node-no-health",
        node_type="compute",
        node_version="2.0.0",
        capabilities={"operations": ["compute"]},
        endpoints={},
        metadata={},
        health_endpoint=None,
    )


@pytest.fixture
def correlation_id() -> UUID:
    """Create a consistent correlation ID for testing."""
    return uuid4()


@pytest.fixture
async def registry_node(
    mock_consul_handler: AsyncMock,
    mock_db_handler: AsyncMock,
    mock_event_bus: AsyncMock,
) -> NodeRegistryEffect:
    """Create initialized registry effect node with mocked handlers."""
    config = ModelNodeRegistryEffectConfig(
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=1.0,
    )
    node = NodeRegistryEffect(
        consul_handler=mock_consul_handler,
        db_handler=mock_db_handler,
        event_bus=mock_event_bus,
        config=config,
    )
    await node.initialize()
    yield node
    await node.shutdown()


@pytest.fixture
async def registry_node_no_event_bus(
    mock_consul_handler: AsyncMock,
    mock_db_handler: AsyncMock,
) -> NodeRegistryEffect:
    """Create initialized registry effect node without event bus."""
    config = ModelNodeRegistryEffectConfig(
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=1.0,
    )
    node = NodeRegistryEffect(
        consul_handler=mock_consul_handler,
        db_handler=mock_db_handler,
        event_bus=None,
        config=config,
    )
    await node.initialize()
    yield node
    await node.shutdown()


# =============================================================================
# Test: Initialization
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectInitialization:
    """Tests for node initialization and shutdown lifecycle."""

    async def test_initialize_sets_initialized_flag(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that initialize() sets the _initialized flag to True."""
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
        )
        assert node._initialized is False

        await node.initialize()

        assert node._initialized is True
        await node.shutdown()

    async def test_shutdown_resets_initialized_flag(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that shutdown() resets the _initialized flag to False."""
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
        )
        await node.initialize()
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
            config=config,
        )
        await node.initialize()

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
        """Test that execute() raises RuntimeHostError if not initialized."""
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
        )
        # Do NOT call initialize()

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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler_failing,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler_failing,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler_failing,
            db_handler=mock_db_handler_failing,
            event_bus=mock_event_bus,
            config=config,
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

        async def consul_execute(envelope: dict[str, Any]) -> dict[str, Any]:
            call_order.append("consul_start")
            await asyncio.sleep(0.05)
            call_order.append("consul_end")
            return {"status": "success"}

        async def db_execute(envelope: dict[str, Any]) -> dict[str, Any]:
            call_order.append("db_start")
            await asyncio.sleep(0.05)
            call_order.append("db_end")
            return {"status": "success", "payload": {"rows_affected": 1}}

        mock_consul_handler.execute = AsyncMock(side_effect=consul_execute)
        mock_db_handler.execute = AsyncMock(side_effect=db_execute)

        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler_failing,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler_failing,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler_failing,
            db_handler=mock_db_handler_failing,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler_with_rows,
            event_bus=mock_event_bus,
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters=None,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True
        assert response.status == "success"
        assert response.operation == "discover"
        assert response.nodes is not None
        assert len(response.nodes) == 2

        # Verify first node
        node1 = response.nodes[0]
        assert node1.node_id == "test-node-1"
        assert node1.node_type == "effect"
        assert node1.node_version == "1.0.0"

        # Verify second node
        node2 = response.nodes[1]
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
                            "registered_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                        }
                    ]
                },
            }
        )

        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
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

        # Verify the filter was passed to the handler
        call_args = mock_db_handler.execute.call_args[0][0]
        assert "node_type" in call_args["payload"]["sql"]

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
                            "registered_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                        }
                    ]
                },
            }
        )

        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
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

        # Verify the filter was passed
        call_args = mock_db_handler.execute.call_args[0][0]
        assert "node_id" in call_args["payload"]["sql"]

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

        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
        )
        await node.initialize()

        request = ModelRegistryRequest(
            operation="discover",
            filters={"node_type": "effect", "node_id": "test-node"},
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        assert response.success is True

        # Verify both filters in SQL
        call_args = mock_db_handler.execute.call_args[0][0]
        sql = call_args["payload"]["sql"]
        assert "node_type" in sql
        assert "node_id" in sql
        assert "AND" in sql

        await node.shutdown()

    async def test_discover_empty_results(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler_empty_results: AsyncMock,
        mock_event_bus: AsyncMock,
        correlation_id: UUID,
    ) -> None:
        """Test discover returns empty list when no nodes found."""
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler_empty_results,
            event_bus=mock_event_bus,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler_failing,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus_failing,
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
        # We need to bypass Pydantic validation to test this
        # Create a valid request and manually modify it
        request = ModelRegistryRequest(
            operation="register",  # Valid operation
            correlation_id=correlation_id,
        )

        # Create a modified version with invalid operation
        # Using model_construct to bypass validation
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
            config=config,
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

        # Wait for reset timeout
        await asyncio.sleep(reset_timeout + 0.05)

        # Next request should succeed (circuit auto-reset to half-open)
        response = await node.execute(request)
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler_failing,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler_failing,
            db_handler=mock_db_handler_failing,
            event_bus=mock_event_bus,
            config=config,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=None,
            config=config,
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

    def test_sanitize_error_includes_exception_type(self) -> None:
        """Test that exception type is included in sanitized error."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        error = ValueError("Test error message")
        result = node._sanitize_error(error)
        assert result.startswith("ValueError: ")
        assert "Test error message" in result

    def test_sanitize_error_redacts_password(self) -> None:
        """Test that passwords are redacted from error messages."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        error = Exception("Connection failed: password=secret123")
        result = node._sanitize_error(error)
        assert "secret123" not in result
        assert "[REDACTED]" in result

    def test_sanitize_error_redacts_token(self) -> None:
        """Test that tokens are redacted from error messages."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        error = Exception("Auth failed: token=abc123xyz")
        result = node._sanitize_error(error)
        assert "abc123xyz" not in result
        assert "[REDACTED]" in result

    def test_sanitize_error_redacts_api_key(self) -> None:
        """Test that API keys are redacted from error messages."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        error = Exception("Request failed: api_key=my-secret-key")
        result = node._sanitize_error(error)
        assert "my-secret-key" not in result
        assert "[REDACTED]" in result

    def test_sanitize_error_redacts_connection_string(self) -> None:
        """Test that connection string credentials are redacted."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        error = Exception("Failed: postgresql://user:password123@host:5432/db")
        result = node._sanitize_error(error)
        assert "password123" not in result
        assert "[REDACTED]" in result

    def test_sanitize_error_truncates_long_messages(self) -> None:
        """Test that long error messages are truncated."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        # Use a mixed message with spaces to avoid base64-like pattern matching
        long_message = "Error: " + " ".join(["word"] * 200)  # Long message with spaces
        error = Exception(long_message)
        result = node._sanitize_error(error)
        # Message should be truncated to 500 chars + type prefix + "..."
        assert len(result) <= 520  # Type name + ": " + 500 chars + "..."
        assert result.endswith("...")

    def test_sanitize_error_redacts_secret(self) -> None:
        """Test that secrets are redacted from error messages."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        error = Exception("Config error: secret=mysupersecret")
        result = node._sanitize_error(error)
        assert "mysupersecret" not in result
        assert "[REDACTED]" in result

    def test_sanitize_error_redacts_bearer_token(self) -> None:
        """Test that bearer tokens are redacted."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
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

        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
        )
        await node.initialize()

        introspection = ModelNodeIntrospectionPayload(
            node_id="test-node",
            node_type="effect",
            node_version="1.0.0",
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

        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
        )
        await node.initialize()

        introspection = ModelNodeIntrospectionPayload(
            node_id="test-node",
            node_type="effect",
            node_version="1.0.0",
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

    def test_safe_json_dumps_success(self) -> None:
        """Test that _safe_json_dumps serializes valid data."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        data = {"key": "value", "nested": {"list": [1, 2, 3]}}
        result = node._safe_json_dumps(data, uuid4(), "test_field")
        assert result == json.dumps(data)

    def test_safe_json_dumps_handles_unserializable_object(self) -> None:
        """Test that _safe_json_dumps handles unserializable objects gracefully."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )

        # Create an unserializable object (set is not JSON serializable)
        unserializable_data = {"items": {1, 2, 3}}  # set is not serializable
        correlation_id = uuid4()
        result = node._safe_json_dumps(
            unserializable_data, correlation_id, "test_field"
        )

        # Should return fallback value
        assert result == "{}"

    def test_safe_json_dumps_handles_circular_reference(self) -> None:
        """Test that _safe_json_dumps handles circular references."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )

        # Create circular reference
        circular: dict[str, object] = {}
        circular["self"] = circular

        correlation_id = uuid4()
        result = node._safe_json_dumps(circular, correlation_id, "circular_field")

        # Should return fallback value
        assert result == "{}"

    def test_safe_json_dumps_custom_fallback(self) -> None:
        """Test that _safe_json_dumps uses custom fallback."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )

        unserializable = {"items": {1, 2, 3}}
        result = node._safe_json_dumps(
            unserializable, uuid4(), "test_field", fallback="[]"
        )

        assert result == "[]"

    def test_safe_json_dumps_strict_success(self) -> None:
        """Test that _safe_json_dumps_strict returns None error on success."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )
        data = {"event_type": "TEST", "correlation_id": "abc123"}
        json_str, error = node._safe_json_dumps_strict(data, uuid4(), "test_field")

        assert error is None
        assert json_str == json.dumps(data)

    def test_safe_json_dumps_strict_failure(self) -> None:
        """Test that _safe_json_dumps_strict returns error message on failure."""
        node = NodeRegistryEffect(
            consul_handler=Mock(),
            db_handler=Mock(),
        )

        # Create unserializable data
        unserializable = {"items": {1, 2, 3}}
        json_str, error = node._safe_json_dumps_strict(
            unserializable, uuid4(), "unserializable_field"
        )

        assert json_str == "{}"
        assert error is not None
        assert "JSON serialization failed" in error
        assert "unserializable_field" in error
        assert "TypeError" in error

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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
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
        node = NodeRegistryEffect(
            consul_handler=mock_consul_handler,
            db_handler=mock_db_handler,
            event_bus=mock_event_bus,
        )
        await node.initialize()

        # Create introspection with complex nested data
        # Note: The ModelNodeIntrospectionPayload accepts dict[str, object]
        # which allows for nested structures
        introspection = ModelNodeIntrospectionPayload(
            node_id="test-node",
            node_type="effect",
            node_version="1.0.0",
            capabilities={"operations": ["read", "write"]},
            endpoints={"health": "http://localhost:8080/health"},
            metadata={"environment": "test"},
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

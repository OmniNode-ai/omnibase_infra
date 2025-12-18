# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for NodeRegistryEffect initialization behavior.

This module tests the new initialize() behavior where dependencies are
automatically resolved if not already done, making the node more resilient
to incorrect usage patterns.

Tests verify:
    - initialize() calls _resolve_dependencies() if not already resolved
    - initialize() is idempotent (safe to call multiple times)
    - initialize() skips resolution if dependencies already resolved
    - create() factory returns a fully initialized node
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from omnibase_core.models.node_metadata import ModelNodeCapabilitiesInfo

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models import (
    ModelNodeIntrospectionPayload,
    ModelNodeRegistryEffectConfig,
    ModelRegistryRequest,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models.model_node_registration_metadata import (
    EnumEnvironment,
    ModelNodeRegistrationMetadata,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.node import NodeRegistryEffect


def create_mock_container(
    consul_handler: AsyncMock,
    db_handler: AsyncMock,
    event_bus: AsyncMock | None = None,
) -> Mock:
    """Create mock ONEX container configured for NodeRegistryEffect.

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
            if name == "postgres":
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
def mock_consul_handler() -> AsyncMock:
    """Create mock Consul handler with successful execute response."""
    handler = AsyncMock()
    handler.execute = AsyncMock(return_value={"status": "success"})
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
def mock_event_bus() -> AsyncMock:
    """Create mock event bus with successful publish."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_container(
    mock_consul_handler: AsyncMock,
    mock_db_handler: AsyncMock,
    mock_event_bus: AsyncMock,
) -> Mock:
    """Create mock container with all handlers registered."""
    return create_mock_container(mock_consul_handler, mock_db_handler, mock_event_bus)


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


# =============================================================================
# Test: Initialize resolves dependencies
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectInitializeResolvesDependencies:
    """Tests for initialize() automatically resolving dependencies.

    These tests verify the robustness improvement where initialize()
    calls _resolve_dependencies() if not already resolved.
    """

    async def test_initialize_sets_initialized_flag(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that initialize() sets the _initialized flag to True."""
        # Create node without initialize() by constructing directly
        node = NodeRegistryEffect(mock_container)
        assert node._initialized is False
        assert node._dependencies_resolved is False

        await node.initialize()

        assert node._initialized is True
        assert node._dependencies_resolved is True
        await node.shutdown()

    async def test_initialize_resolves_dependencies_if_not_resolved(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that initialize() calls _resolve_dependencies() if not already resolved.

        This is the key behavior added for robustness - users who forget to call
        _resolve_dependencies() before initialize() will still have a working node.
        """
        node = NodeRegistryEffect(mock_container)

        # Dependencies should not be resolved yet
        assert node._dependencies_resolved is False
        assert node._consul_handler is None
        assert node._db_handler is None

        # Call initialize() without calling _resolve_dependencies() first
        await node.initialize()

        # Dependencies should now be resolved
        assert node._dependencies_resolved is True
        assert node._consul_handler is not None
        assert node._db_handler is not None
        assert node._initialized is True

        await node.shutdown()

    async def test_initialize_is_idempotent(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that calling initialize() multiple times is safe (idempotent)."""
        node = NodeRegistryEffect(mock_container)

        # Call initialize multiple times
        await node.initialize()
        await node.initialize()
        await node.initialize()

        # Should still be initialized
        assert node._initialized is True
        assert node._dependencies_resolved is True

        await node.shutdown()

    async def test_initialize_skips_resolve_if_already_resolved(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that initialize() skips dependency resolution if already done."""
        node = NodeRegistryEffect(mock_container)

        # Manually resolve dependencies first
        await node._resolve_dependencies()
        assert node._dependencies_resolved is True

        # Clear the mock call history
        mock_container.service_registry.resolve_service.reset_mock()

        # Call initialize()
        await node.initialize()

        # resolve_service should NOT have been called again
        mock_container.service_registry.resolve_service.assert_not_called()
        assert node._initialized is True

        await node.shutdown()

    async def test_create_factory_returns_fully_initialized_node(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that create() factory method returns a fully initialized node."""
        node = await NodeRegistryEffect.create(mock_container)

        # Node should be fully initialized and ready to use
        assert node._initialized is True
        assert node._dependencies_resolved is True
        assert node._consul_handler is not None
        assert node._db_handler is not None

        await node.shutdown()

    async def test_execute_raises_when_not_initialized(
        self,
        mock_container: Mock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that execute() raises RuntimeHostError if not initialized.

        Even with dependencies resolved, execute() requires initialize() to be called.
        """
        node = NodeRegistryEffect(mock_container)
        # Resolve dependencies but do NOT call initialize()
        await node._resolve_dependencies()
        assert node._dependencies_resolved is True
        assert node._initialized is False

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        assert "not initialized" in exc_info.value.message.lower()

    async def test_node_works_correctly_after_initialize_resolves_dependencies(
        self,
        mock_container: Mock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that node works correctly when initialize() resolves dependencies.

        This is the integration test that verifies the whole flow works:
        1. Create node (no dependencies resolved)
        2. Call initialize() (automatically resolves dependencies)
        3. Execute operation (should succeed)
        """
        node = NodeRegistryEffect(mock_container)

        # Initially not resolved
        assert node._dependencies_resolved is False
        assert node._initialized is False

        # Initialize (should resolve dependencies automatically)
        await node.initialize()

        # Now execute should work
        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        response = await node.execute(request)

        # Operation should succeed
        assert response.success is True
        assert response.status == "success"
        assert response.operation == "register"

        await node.shutdown()

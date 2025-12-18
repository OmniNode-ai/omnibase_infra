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

    async def test_initialize_returns_early_when_already_initialized(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that initialize() returns early without re-resolving when already initialized.

        This test verifies the idempotency optimization where a second call to
        initialize() returns immediately without re-resolving dependencies.
        """
        node = NodeRegistryEffect(mock_container)

        # First initialize
        await node.initialize()
        assert node._initialized is True

        # Record call count after first initialize
        resolve_call_count_after_first = (
            mock_container.service_registry.resolve_service.call_count
        )

        # Second initialize should return early
        await node.initialize()

        # resolve_service should NOT have been called again (early return)
        assert (
            mock_container.service_registry.resolve_service.call_count
            == resolve_call_count_after_first
        ), "resolve_service should not be called on second initialize()"

        # Node should still be initialized
        assert node._initialized is True
        assert node._dependencies_resolved is True

        await node.shutdown()


# =============================================================================
# Test: _ensure_dependencies() error paths
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectEnsureDependencies:
    """Tests for _ensure_dependencies() error paths.

    These tests verify proper error handling when:
    - Dependencies haven't been resolved yet
    - Required handlers are None after resolution
    - Accessing consul_handler/db_handler properties before resolution
    """

    async def test_ensure_dependencies_raises_when_not_resolved(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that _ensure_dependencies() raises RuntimeError when not resolved.

        This verifies the first error path: dependencies_resolved is False.
        """
        node = NodeRegistryEffect(mock_container)

        # Dependencies not yet resolved
        assert node._dependencies_resolved is False

        with pytest.raises(RuntimeError) as exc_info:
            node._ensure_dependencies()

        error_message = str(exc_info.value)
        assert "Dependencies not resolved" in error_message
        assert "_resolve_dependencies()" in error_message or "create()" in error_message

    async def test_consul_handler_property_raises_when_not_resolved(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that consul_handler property raises when dependencies not resolved."""
        node = NodeRegistryEffect(mock_container)
        assert node._dependencies_resolved is False

        with pytest.raises(RuntimeError) as exc_info:
            _ = node.consul_handler

        assert "Dependencies not resolved" in str(exc_info.value)

    async def test_db_handler_property_raises_when_not_resolved(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that db_handler property raises when dependencies not resolved."""
        node = NodeRegistryEffect(mock_container)
        assert node._dependencies_resolved is False

        with pytest.raises(RuntimeError) as exc_info:
            _ = node.db_handler

        assert "Dependencies not resolved" in str(exc_info.value)

    async def test_ensure_dependencies_raises_when_consul_handler_none(
        self,
    ) -> None:
        """Test that _ensure_dependencies() raises when consul_handler is None.

        This tests the edge case where dependencies_resolved is True but
        the consul_handler is still None (should not happen in normal flow).
        """
        # Create a mock container that returns None for consul
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        # Mock that sets None for consul handler
        async def resolve_service_returning_none(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock | None:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return None  # Return None to simulate edge case
                if name == "postgres":
                    handler = AsyncMock()
                    handler.execute = AsyncMock(return_value={"status": "success"})
                    return handler
            if interface_type is ProtocolEventBus:
                raise ValueError("Not registered")
            return None

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_returning_none
        )

        node = NodeRegistryEffect(container)

        # Manually simulate partial resolution (edge case scenario)
        node._dependencies_resolved = True
        node._consul_handler = None
        node._db_handler = AsyncMock()

        with pytest.raises(RuntimeError) as exc_info:
            node._ensure_dependencies()

        error_message = str(exc_info.value)
        assert "Required handlers" in error_message
        assert "consul_handler" in error_message or "db_handler" in error_message

    async def test_ensure_dependencies_raises_when_db_handler_none(
        self,
    ) -> None:
        """Test that _ensure_dependencies() raises when db_handler is None.

        This tests the edge case where dependencies_resolved is True but
        the db_handler is still None (should not happen in normal flow).
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_service_returning_none(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock | None:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    handler = AsyncMock()
                    handler.execute = AsyncMock(return_value={"status": "success"})
                    return handler
                if name == "postgres":
                    return None  # Return None to simulate edge case
            if interface_type is ProtocolEventBus:
                raise ValueError("Not registered")
            return None

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_returning_none
        )

        node = NodeRegistryEffect(container)

        # Manually simulate partial resolution (edge case scenario)
        node._dependencies_resolved = True
        node._consul_handler = AsyncMock()
        node._db_handler = None

        with pytest.raises(RuntimeError) as exc_info:
            node._ensure_dependencies()

        error_message = str(exc_info.value)
        assert "Required handlers" in error_message

    async def test_ensure_dependencies_passes_when_properly_resolved(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that _ensure_dependencies() passes when properly resolved."""
        node = NodeRegistryEffect(mock_container)
        await node._resolve_dependencies()

        # Should not raise
        node._ensure_dependencies()

        assert node._dependencies_resolved is True
        assert node._consul_handler is not None
        assert node._db_handler is not None


# =============================================================================
# Test: _resolve_required_handler() error paths
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectResolveRequiredHandler:
    """Tests for _resolve_required_handler() error paths.

    These tests verify proper error handling when:
    - Service is not registered (KeyError, LookupError, ServiceResolutionError)
    - Configuration/validation error (ValueError, TypeError)
    - Unexpected error (generic Exception)
    """

    async def test_consul_handler_not_registered_raises_runtime_error(
        self,
    ) -> None:
        """Test that RuntimeError is raised when consul handler not registered.

        Verifies the KeyError/LookupError/ServiceResolutionError path.
        """
        from omnibase_infra.errors import ServiceResolutionError
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_service_consul_not_found(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    raise KeyError("Service 'consul' not registered")
                if name == "postgres":
                    handler = AsyncMock()
                    handler.execute = AsyncMock(return_value={"status": "success"})
                    return handler
            if interface_type is ProtocolEventBus:
                raise ValueError("Not registered")
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_consul_not_found
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "Failed to resolve" in error_message
        assert "Consul handler" in error_message
        assert "consul" in error_message
        assert "KeyError" in error_message

    async def test_postgres_handler_not_registered_raises_runtime_error(
        self,
    ) -> None:
        """Test that RuntimeError is raised when postgres handler not registered.

        Verifies the KeyError/LookupError/ServiceResolutionError path.
        """
        from omnibase_infra.errors import ServiceResolutionError
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_service_postgres_not_found(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    handler = AsyncMock()
                    handler.execute = AsyncMock(return_value={"status": "success"})
                    return handler
                if name == "postgres":
                    raise LookupError("Service 'postgres' not found in registry")
            if interface_type is ProtocolEventBus:
                raise ValueError("Not registered")
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_postgres_not_found
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "Failed to resolve" in error_message
        assert "PostgreSQL handler" in error_message
        assert "postgres" in error_message
        assert "LookupError" in error_message

    async def test_service_resolution_error_raises_runtime_error(
        self,
    ) -> None:
        """Test that ServiceResolutionError is properly wrapped in RuntimeError."""
        from omnibase_infra.errors import ServiceResolutionError
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_service_resolution_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    raise ServiceResolutionError(
                        "Failed to resolve service: consul handler unavailable"
                    )
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_resolution_error
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "Failed to resolve" in error_message
        assert "ServiceResolutionError" in error_message

    async def test_configuration_error_raises_runtime_error(
        self,
    ) -> None:
        """Test that ValueError during resolution raises RuntimeError.

        Verifies the ValueError/TypeError path for configuration errors.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_service_config_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    raise ValueError(
                        "Invalid configuration: missing required field 'host'"
                    )
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_config_error
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "Configuration error" in error_message
        assert "Consul handler" in error_message
        assert "ValueError" in error_message
        assert "Invalid configuration" in error_message

    async def test_type_error_raises_runtime_error(
        self,
    ) -> None:
        """Test that TypeError during resolution raises RuntimeError.

        Verifies the TypeError path for type mismatch errors.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_service_type_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "postgres":
                    raise TypeError("Expected ProtocolEnvelopeExecutor, got str")
                if name == "consul":
                    handler = AsyncMock()
                    handler.execute = AsyncMock(return_value={"status": "success"})
                    return handler
            if interface_type is ProtocolEventBus:
                raise ValueError("Not registered")
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_type_error
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "Configuration error" in error_message
        assert "PostgreSQL handler" in error_message
        assert "TypeError" in error_message

    async def test_unexpected_error_raises_runtime_error(
        self,
    ) -> None:
        """Test that unexpected exceptions are wrapped in RuntimeError.

        Verifies the generic Exception path for unexpected errors.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_service_unexpected_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    raise OSError("Network unreachable")
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_service_unexpected_error
        )

        node = NodeRegistryEffect(container)

        with pytest.raises(RuntimeError) as exc_info:
            await node._resolve_dependencies()

        error_message = str(exc_info.value)
        assert "Unexpected error" in error_message
        assert "Consul handler" in error_message
        assert "OSError" in error_message
        assert "Network unreachable" in error_message


# =============================================================================
# Test: _resolve_optional_service() error paths
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectResolveOptionalService:
    """Tests for _resolve_optional_service() error paths.

    These tests verify proper handling when optional services fail:
    - Service not registered (returns None, no error)
    - Configuration error (returns None with warning)
    - Unexpected error (returns None with warning)
    """

    async def test_event_bus_not_registered_returns_none(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that missing event bus returns None without error.

        Event bus is optional - missing registration should not raise.
        """
        container = create_mock_container(
            mock_consul_handler, mock_db_handler, event_bus=None
        )

        node = NodeRegistryEffect(container)
        await node._resolve_dependencies()

        # Event bus should be None but node should be usable
        assert node._event_bus is None
        assert node._dependencies_resolved is True
        assert node._consul_handler is not None
        assert node._db_handler is not None

    async def test_event_bus_configuration_error_returns_none(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that ValueError during event bus resolution returns None.

        Configuration errors for optional services should not fail initialization.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_with_event_bus_config_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return mock_consul_handler
                if name == "postgres":
                    return mock_db_handler
            if interface_type is ProtocolEventBus:
                # Configuration error for optional service
                raise ValueError("Invalid Kafka configuration: broker list empty")
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_with_event_bus_config_error
        )

        node = NodeRegistryEffect(container)
        await node._resolve_dependencies()

        # Event bus should be None due to config error
        assert node._event_bus is None
        assert node._dependencies_resolved is True
        # Required handlers should still be resolved
        assert node._consul_handler is not None
        assert node._db_handler is not None

    async def test_event_bus_type_error_returns_none(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that TypeError during event bus resolution returns None."""
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_with_event_bus_type_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return mock_consul_handler
                if name == "postgres":
                    return mock_db_handler
            if interface_type is ProtocolEventBus:
                raise TypeError("Expected ProtocolEventBus, got MockKafka")
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_with_event_bus_type_error
        )

        node = NodeRegistryEffect(container)
        await node._resolve_dependencies()

        # Event bus should be None due to type error
        assert node._event_bus is None
        assert node._dependencies_resolved is True

    async def test_event_bus_unexpected_error_returns_none(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that unexpected error during event bus resolution returns None.

        Graceful degradation: unexpected errors for optional services should
        not prevent initialization.
        """
        from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
            ProtocolEnvelopeExecutor,
            ProtocolEventBus,
        )

        container = Mock()
        container.service_registry = Mock()

        async def resolve_with_event_bus_unexpected_error(
            interface_type: type,
            name: str | None = None,
        ) -> AsyncMock:
            if interface_type is ProtocolEnvelopeExecutor:
                if name == "consul":
                    return mock_consul_handler
                if name == "postgres":
                    return mock_db_handler
            if interface_type is ProtocolEventBus:
                # Unexpected infrastructure error
                raise OSError("Kafka broker connection refused")
            raise ValueError(f"Unknown service: {interface_type}")

        container.service_registry.resolve_service = AsyncMock(
            side_effect=resolve_with_event_bus_unexpected_error
        )

        node = NodeRegistryEffect(container)
        await node._resolve_dependencies()

        # Event bus should be None due to unexpected error
        assert node._event_bus is None
        assert node._dependencies_resolved is True
        # Required handlers should still be resolved
        assert node._consul_handler is not None
        assert node._db_handler is not None

    async def test_request_introspection_raises_when_event_bus_none(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that request_introspection raises RuntimeHostError when no event bus.

        Even though event bus resolution doesn't fail, operations requiring
        it should raise a clear error.
        """
        container = create_mock_container(
            mock_consul_handler, mock_db_handler, event_bus=None
        )

        node = await NodeRegistryEffect.create(container)

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="request_introspection",
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        error_message = exc_info.value.message.lower()
        assert "event bus not configured" in error_message
        # correlation_id is stored at model level, not in context dict
        assert exc_info.value.model.correlation_id == correlation_id

        await node.shutdown()


# =============================================================================
# Test: Correlation ID propagation in errors
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNodeRegistryEffectCorrelationIdPropagation:
    """Tests for correlation ID propagation in error contexts.

    These tests verify that correlation IDs are properly included in
    error context for distributed tracing.
    """

    async def test_register_missing_introspection_includes_correlation_id(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that register errors include correlation_id in error model."""
        node = await NodeRegistryEffect.create(mock_container)

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=None,  # Missing required field
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        # Verify correlation_id is stored at model level
        assert exc_info.value.model.correlation_id == correlation_id

        await node.shutdown()

    async def test_deregister_missing_node_id_includes_correlation_id(
        self,
        mock_container: Mock,
    ) -> None:
        """Test that deregister errors include correlation_id in error model."""
        node = await NodeRegistryEffect.create(mock_container)

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="deregister",
            node_id=None,  # Missing required field
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        # Verify correlation_id is stored at model level
        assert exc_info.value.model.correlation_id == correlation_id

        await node.shutdown()

    async def test_not_initialized_error_includes_correlation_id(
        self,
        mock_container: Mock,
        introspection_payload: ModelNodeIntrospectionPayload,
    ) -> None:
        """Test that not initialized errors include correlation_id in error model."""
        node = NodeRegistryEffect(mock_container)
        # Resolve dependencies but don't initialize
        await node._resolve_dependencies()

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="register",
            introspection_event=introspection_payload,
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        # Verify correlation_id is stored at model level
        assert exc_info.value.model.correlation_id == correlation_id

    async def test_request_introspection_no_event_bus_includes_correlation_id(
        self,
        mock_consul_handler: AsyncMock,
        mock_db_handler: AsyncMock,
    ) -> None:
        """Test that request_introspection without event bus includes correlation_id."""
        container = create_mock_container(
            mock_consul_handler, mock_db_handler, event_bus=None
        )
        node = await NodeRegistryEffect.create(container)

        correlation_id = uuid4()
        request = ModelRegistryRequest(
            operation="request_introspection",
            correlation_id=correlation_id,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await node.execute(request)

        # Verify correlation_id is stored at model level
        assert exc_info.value.model.correlation_id == correlation_id

        await node.shutdown()

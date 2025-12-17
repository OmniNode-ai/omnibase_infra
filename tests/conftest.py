"""Pytest configuration and shared fixtures for omnibase_infra tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
    from omnibase_infra.runtime.policy_registry import PolicyRegistry


@pytest.fixture
def mock_container() -> MagicMock:
    """Create mock ONEX container for testing.

    Provides a mock ModelONEXContainer with service_registry that supports
    basic resolution and registration patterns. Methods are AsyncMock since
    omnibase_core 0.4.x+ uses async container methods.

    Returns:
        MagicMock configured to mimic ModelONEXContainer API.

    Example:
        >>> async def test_with_container(mock_container):
        ...     # Mock container is ready to use (async methods)
        ...     mock_container.service_registry.resolve_service.return_value = some_service
    """
    from unittest.mock import AsyncMock
    from unittest.mock import MagicMock as SyncMock

    container = MagicMock()

    # Mock legacy methods for backwards compatibility
    container.get_config.return_value = {}

    # Mock service_registry for container-based DI
    # Note: Both resolve_service and register_instance are async in omnibase_core 0.4.x+
    # For integration tests with real containers, use container_with_registries.
    container.service_registry = MagicMock()
    container.service_registry.resolve_service = (
        AsyncMock()
    )  # Async in omnibase_core 0.4+
    container.service_registry.register_instance = AsyncMock(
        return_value="mock-uuid"
    )  # Async for wire functions

    return container


@pytest.fixture
def container_with_policy_registry(mock_container: MagicMock) -> PolicyRegistry:
    """Create PolicyRegistry and configure mock container to resolve it.

    Provides a real PolicyRegistry instance registered in a mock container.
    This fixture demonstrates the container-based DI pattern for testing.

    Args:
        mock_container: Mock container fixture (automatically injected).

    Returns:
        PolicyRegistry instance that can be resolved from mock_container.

    Example:
        >>> def test_container_based_policy_access(container_with_policy_registry, mock_container):
        ...     # Registry is already registered in mock_container
        ...     from omnibase_infra.runtime.policy_registry import PolicyRegistry
        ...     registry = mock_container.service_registry.resolve_service(PolicyRegistry)
        ...     assert registry is container_with_policy_registry
        ...
        ...     # Use registry to register and retrieve policies
        ...     from omnibase_infra.enums import EnumPolicyType
        ...     registry.register_policy(
        ...         policy_id="test_policy",
        ...         policy_class=MockPolicy,
        ...         policy_type=EnumPolicyType.ORCHESTRATOR,
        ...     )
        ...     assert registry.is_registered("test_policy")
    """
    from omnibase_infra.runtime.policy_registry import PolicyRegistry

    # Create real PolicyRegistry instance
    registry = PolicyRegistry()

    # Configure mock container to return this registry when resolved
    async def resolve_service_side_effect(interface_type: type) -> PolicyRegistry:
        if interface_type is PolicyRegistry:
            return registry
        raise ValueError(f"Service not registered: {interface_type}")

    mock_container.service_registry.resolve_service.side_effect = (
        resolve_service_side_effect
    )

    return registry


@pytest.fixture
async def container_with_registries() -> ModelONEXContainer:
    """Create real ONEX container with wired infrastructure services.

    Provides a fully wired ModelONEXContainer with PolicyRegistry and
    ProtocolBindingRegistry registered as global services. This fixture
    demonstrates the real container-based DI pattern for integration tests.

    Note: This fixture is async because wire_infrastructure_services() is async.

    Returns:
        ModelONEXContainer instance with infrastructure services wired.

    Example:
        >>> async def test_with_real_container(container_with_registries):
        ...     from omnibase_infra.runtime.policy_registry import PolicyRegistry
        ...     from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
        ...
        ...     # Resolve services from real container (async)
        ...     policy_reg = await container_with_registries.service_registry.resolve_service(PolicyRegistry)
        ...     handler_reg = await container_with_registries.service_registry.resolve_service(ProtocolBindingRegistry)
        ...
        ...     # Use registries
        ...     assert isinstance(policy_reg, PolicyRegistry)
        ...     assert isinstance(handler_reg, ProtocolBindingRegistry)
    """
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

    # Create real container
    container = ModelONEXContainer()

    # Wire infrastructure services (async operation)
    await wire_infrastructure_services(container)

    return container


@pytest.fixture
async def container_with_handler_registry(
    container_with_registries: ModelONEXContainer,
) -> ProtocolBindingRegistry:
    """Get ProtocolBindingRegistry from wired container.

    Convenience fixture that extracts ProtocolBindingRegistry from the
    container_with_registries fixture. Use this when you only need the
    handler registry without the full container.

    Note: This fixture is async because resolve_service() is async.

    Args:
        container_with_registries: Container fixture (automatically injected).

    Returns:
        ProtocolBindingRegistry instance from container.

    Example:
        >>> async def test_handler_registry(container_with_handler_registry):
        ...     from omnibase_infra.runtime.handler_registry import HANDLER_TYPE_HTTP
        ...     container_with_handler_registry.register(HANDLER_TYPE_HTTP, MockHandler)
        ...     assert container_with_handler_registry.is_registered(HANDLER_TYPE_HTTP)
    """
    from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry

    registry: ProtocolBindingRegistry = (
        await container_with_registries.service_registry.resolve_service(
            ProtocolBindingRegistry
        )
    )
    return registry

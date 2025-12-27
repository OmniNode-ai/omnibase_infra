"""Pytest configuration and shared fixtures for omnibase_infra tests."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
    from omnibase_infra.runtime.policy_registry import PolicyRegistry


# =============================================================================
# Duck Typing Conformance Helpers
# =============================================================================


def assert_has_methods(
    obj: object,
    required_methods: list[str],
    *,
    protocol_name: str | None = None,
) -> None:
    """Assert that an object has all required methods (duck typing conformance).

    Per ONEX conventions, protocol conformance is verified via duck typing
    by checking for required method presence and callability, rather than
    using isinstance checks with Protocol types.

    Args:
        obj: The object to check for method presence.
        required_methods: List of method names that must be present and callable.
        protocol_name: Optional protocol name for clearer error messages.

    Raises:
        AssertionError: If any required method is missing or not callable.

    Example:
        >>> assert_has_methods(
        ...     registry,
        ...     ["register", "get", "list_keys", "is_registered"],
        ...     protocol_name="PolicyRegistry",
        ... )
    """
    name = protocol_name or obj.__class__.__name__
    for method_name in required_methods:
        assert hasattr(obj, method_name), f"{name} must have '{method_name}' method"
        # __len__ and __iter__ are special - they are callable via len()/iter()
        if not method_name.startswith("__"):
            assert callable(
                getattr(obj, method_name)
            ), f"{name}.{method_name} must be callable"


def assert_has_async_methods(
    obj: object,
    required_methods: list[str],
    *,
    protocol_name: str | None = None,
) -> None:
    """Assert that an object has all required async methods.

    Extended duck typing verification that also checks that methods are
    coroutine functions (async).

    Args:
        obj: The object to check for async method presence.
        required_methods: List of method names that must be async and callable.
        protocol_name: Optional protocol name for clearer error messages.

    Raises:
        AssertionError: If any method is missing, not callable, or not async.

    Example:
        >>> assert_has_async_methods(
        ...     reducer,
        ...     ["reduce"],
        ...     protocol_name="ProtocolReducer",
        ... )
    """
    name = protocol_name or obj.__class__.__name__
    for method_name in required_methods:
        assert hasattr(obj, method_name), f"{name} must have '{method_name}' method"
        method = getattr(obj, method_name)
        assert callable(method), f"{name}.{method_name} must be callable"
        assert asyncio.iscoroutinefunction(
            method
        ), f"{name}.{method_name} must be async (coroutine function)"


def assert_method_signature(
    obj: object,
    method_name: str,
    expected_params: list[str],
    *,
    protocol_name: str | None = None,
) -> None:
    """Assert that a method has the expected parameter signature.

    Verifies that a method's signature contains the expected parameters.
    Does not check parameter types, only names.

    Args:
        obj: The object containing the method.
        method_name: Name of the method to check.
        expected_params: List of expected parameter names (excluding 'self').
        protocol_name: Optional protocol name for clearer error messages.

    Raises:
        AssertionError: If method is missing or parameters don't match.

    Example:
        >>> assert_method_signature(
        ...     reducer,
        ...     "reduce",
        ...     ["state", "event"],
        ...     protocol_name="ProtocolReducer",
        ... )
    """
    name = protocol_name or obj.__class__.__name__
    assert hasattr(obj, method_name), f"{name} must have '{method_name}' method"

    method = getattr(obj, method_name)
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    assert len(params) == len(expected_params), (
        f"{name}.{method_name} must have {len(expected_params)} parameters "
        f"({', '.join(expected_params)}), got {len(params)}: {params}"
    )

    for expected in expected_params:
        assert (
            expected in params
        ), f"{name}.{method_name} must have '{expected}' parameter, got: {params}"


# =============================================================================
# Registry-Specific Conformance Helpers
# =============================================================================


def assert_policy_registry_interface(registry: object) -> None:
    """Assert that an object implements the PolicyRegistry interface.

    Per ONEX conventions, protocol conformance is verified via duck typing.
    Collection-like protocols must include __len__ for complete duck typing.

    Args:
        registry: The object to verify as a PolicyRegistry implementation.

    Raises:
        AssertionError: If required methods are missing.

    Example:
        >>> registry = await get_policy_registry_from_container(container)
        >>> assert_policy_registry_interface(registry)
        >>> assert len(registry) == 0  # Empty initially
    """
    required_methods = [
        "register",
        "register_policy",
        "get",
        "list_keys",
        "is_registered",
        "__len__",
    ]
    assert_has_methods(registry, required_methods, protocol_name="PolicyRegistry")


def assert_handler_registry_interface(registry: object) -> None:
    """Assert that an object implements the ProtocolBindingRegistry interface.

    Per ONEX conventions, protocol conformance is verified via duck typing.
    Collection-like protocols must include __len__ for complete duck typing.

    Args:
        registry: The object to verify as a ProtocolBindingRegistry implementation.

    Raises:
        AssertionError: If required methods are missing.

    Example:
        >>> registry = await get_handler_registry_from_container(container)
        >>> assert_handler_registry_interface(registry)
        >>> assert len(registry) == 0
    """
    required_methods = [
        "register",
        "get",
        "list_protocols",
        "is_registered",
        "__len__",
    ]
    assert_has_methods(
        registry, required_methods, protocol_name="ProtocolBindingRegistry"
    )


def assert_reducer_protocol_interface(reducer: object) -> None:
    """Assert that an object implements the ProtocolReducer interface.

    Verifies that the reducer has the required async reduce() method with
    the correct signature (state, event).

    Args:
        reducer: The object to verify as a ProtocolReducer implementation.

    Raises:
        AssertionError: If required methods/signatures don't match.

    Example:
        >>> assert_reducer_protocol_interface(mock_reducer)
    """
    assert_has_async_methods(reducer, ["reduce"], protocol_name="ProtocolReducer")
    assert_method_signature(
        reducer, "reduce", ["state", "event"], protocol_name="ProtocolReducer"
    )


def assert_effect_protocol_interface(effect: object) -> None:
    """Assert that an object implements the ProtocolEffect interface.

    Verifies that the effect has the required async execute_intent() method
    with the correct signature (intent, correlation_id).

    Args:
        effect: The object to verify as a ProtocolEffect implementation.

    Raises:
        AssertionError: If required methods/signatures don't match.

    Example:
        >>> assert_effect_protocol_interface(mock_effect)
    """
    assert_has_async_methods(effect, ["execute_intent"], protocol_name="ProtocolEffect")
    assert_method_signature(
        effect,
        "execute_intent",
        ["intent", "correlation_id"],
        protocol_name="ProtocolEffect",
    )


def assert_dispatcher_protocol_interface(dispatcher: object) -> None:
    """Assert that an object implements the ProtocolMessageDispatcher interface.

    Verifies that the dispatcher has all required properties and methods.

    Args:
        dispatcher: The object to verify as a ProtocolMessageDispatcher.

    Raises:
        AssertionError: If required properties/methods are missing.

    Example:
        >>> assert_dispatcher_protocol_interface(my_dispatcher)
    """
    required_props = ["dispatcher_id", "category", "message_types", "node_kind"]
    for prop in required_props:
        assert hasattr(
            dispatcher, prop
        ), f"ProtocolMessageDispatcher must have '{prop}' property"

    assert hasattr(
        dispatcher, "handle"
    ), "ProtocolMessageDispatcher must have 'handle' method"
    assert callable(
        dispatcher.handle
    ), "ProtocolMessageDispatcher.handle must be callable"


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
async def container_with_registries() -> AsyncGenerator[ModelONEXContainer, None]:
    """Create real ONEX container with wired infrastructure services.

    Provides a fully wired ModelONEXContainer with PolicyRegistry and
    ProtocolBindingRegistry registered as global services. This fixture
    demonstrates the real container-based DI pattern for integration tests.

    Note: This fixture is async because wire_infrastructure_services() is async.

    Yields:
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
        ...     # Verify interface via duck typing (ONEX convention)
        ...     # Per ONEX conventions, check for required methods rather than isinstance
        ...     assert hasattr(policy_reg, "register_policy")
        ...     assert hasattr(handler_reg, "register")
    """
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

    # Create real container
    container = ModelONEXContainer()

    # Wire infrastructure services (async operation)
    await wire_infrastructure_services(container)

    return container

    # Cleanup: ModelONEXContainer doesn't have explicit cleanup methods,
    # but using yield ensures proper fixture teardown semantics and allows
    # for future cleanup needs.


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

"""Pytest configuration and shared fixtures for omnibase_infra tests.

==============================================================================
IMPORTANT: Event Loop Scope Configuration (pytest-asyncio 0.25+)
==============================================================================

This module provides session-scoped and function-scoped async fixtures. With
pytest-asyncio 0.25+, the default event loop scope changed from "session" to
"function", which can cause "attached to a different loop" errors when sharing
async resources across tests.

Global Configuration (pyproject.toml)
-------------------------------------
This project uses ``asyncio_mode = "auto"`` in ``[tool.pytest.ini_options]``.
This auto-detects async tests but does NOT set a global loop scope.

When to Configure loop_scope
----------------------------
**Test modules that use session-scoped async fixtures** must explicitly set
the loop_scope via pytestmark:

.. code-block:: python

    # For session-scoped fixtures (shared across entire test session)
    pytestmark = [pytest.mark.asyncio(loop_scope="session")]

    # For module-scoped fixtures (shared within a single test module)
    pytestmark = [pytest.mark.asyncio(loop_scope="module")]

    # For function-scoped fixtures only (default - no config needed)
    # Each test gets its own event loop

Why This Matters
----------------
- **Session/Module-scoped async fixtures**: Require matching loop_scope to
  share async resources (database connections, Kafka producers, etc.)
- **Function-scoped async fixtures**: Work with default settings (each test
  gets isolated event loop)
- **RuntimeError symptoms**: "attached to a different loop" or "Event loop is
  closed" typically indicate loop_scope mismatch

Fixture Scope Reference
-----------------------
This module provides the following async fixtures:

Session-scoped (require loop_scope="session" in test modules):
    - (none currently - add here if session-scoped fixtures are added)

Function-scoped (work with default settings):
    - event_bus: In-memory event bus for testing
    - container_with_registries: Real ONEX container with wired services

Reference Documentation
-----------------------
- https://pytest-asyncio.readthedocs.io/en/latest/concepts.html#event-loop-scope
- https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/change_default_loop_scope.html

Related Tickets:
    - OMN-1361: pytest-asyncio 0.25+ upgrade and loop_scope configuration
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file at test session start
# This enables tests to use infrastructure config (CONSUL_HOST, KAFKA_BOOTSTRAP_SERVERS, etc.)
# without needing to set env vars on command line
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
    logging.getLogger(__name__).debug(f"Loaded environment from {_env_file}")

from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.utils import sanitize_error_message

# Module-level logger for test cleanup diagnostics
logger = logging.getLogger(__name__)


# =============================================================================
# Test Node Identity Helper
# =============================================================================


def make_test_node_identity(
    suffix: str = "",
    *,
    env: str = "test",
    service: str = "test-service",
    node_name: str = "test-node",
    version: str = "v1",
) -> ModelNodeIdentity:
    """Create a test node identity for subscribe() calls.

    This is the consolidated helper for creating ModelNodeIdentity instances
    in unit and integration tests. For E2E tests with specialized documentation,
    use the helper in tests/integration/registration/e2e/conftest.py.

    Args:
        suffix: Optional suffix to differentiate identities for tests
               that need multiple distinct consumer groups.
        env: Environment name (default: "test").
        service: Service name (default: "test-service").
        node_name: Node name base (default: "test-node").
        version: Version string (default: "v1").

    Returns:
        A ModelNodeIdentity configured for testing.

    Example:
        >>> identity = make_test_node_identity()
        >>> identity.node_name
        'test-node'

        >>> identity = make_test_node_identity("subscriber-1")
        >>> identity.node_name
        'test-node-subscriber-1'

        >>> identity = make_test_node_identity(service="kafka-tests")
        >>> identity.service
        'kafka-tests'

    Note:
        used for consumer group derivation, but it's still required by the
        subscribe() signature.

    .. versionadded:: 0.2.7
        Consolidated from duplicate helpers across test modules (OMN-1602).
    """
    actual_node_name = f"{node_name}-{suffix}" if suffix else node_name
    return ModelNodeIdentity(
        env=env, service=service, node_name=actual_node_name, version=version
    )


if TYPE_CHECKING:
    # TYPE_CHECKING imports: These imports are only used for type annotations.
    # They are NOT imported at runtime, which:
    # 1. Avoids circular import issues (container modules may import test utilities)
    # 2. Reduces import overhead during test collection
    # 3. Prevents runtime errors if these modules have heavy dependencies
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
    from omnibase_infra.runtime.registry_policy import RegistryPolicy


# =============================================================================
# Service Registry Availability Check
# =============================================================================


def check_service_registry_available() -> bool:
    """Check if ServiceRegistry is available in ModelONEXContainer.

    Creates a temporary container to check for service_registry availability,
    then explicitly cleans up the container to prevent resource leaks.

    This function is used by test modules to determine whether to skip tests
    that require ServiceRegistry. The check is needed because omnibase_core 0.6.x
    has a circular import issue that causes ServiceRegistry to be None when
    the container is initialized.

    Returns:
        True if service_registry is available and not None, False otherwise.

    Note:
        The circular import path in omnibase_core 0.6.2 is:
        model_onex_container.py -> container_service_registry.py ->
        container/__init__.py -> container_service_resolver.py ->
        ModelONEXContainer (still loading)

        Tests requiring ServiceRegistry should skip gracefully when this
        function returns False. Upgrade to omnibase_core >= 0.6.3 to resolve.
    """
    container = None
    try:
        from omnibase_core.container import ModelONEXContainer

        container = ModelONEXContainer()
        return container.service_registry is not None
    except AttributeError:
        # service_registry attribute removed in omnibase_core 0.6.x
        return False
    except TypeError:
        # ModelONEXContainer.__init__ signature changed (new required params)
        return False
    except ImportError:
        # omnibase_core not installed or import failed
        return False
    finally:
        # Explicit cleanup of temporary container
        del container


# =============================================================================
# Duck Typing Conformance Helpers
# =============================================================================
# Protocol Compliance Strategy:
#
# ONEX uses TWO complementary approaches for protocol conformance:
#
# 1. Duck Typing Helpers (assert_has_methods, assert_has_async_methods):
#    - Check method presence and callability at runtime
#    - Work with ANY object, regardless of explicit Protocol inheritance
#    - Preferred for verifying expected interface contracts
#    - No dependency on @runtime_checkable decorator
#
# 2. isinstance() with @runtime_checkable Protocols:
#    - Used when protocols are marked with @runtime_checkable
#    - Provides IDE autocompletion and type checker support
#    - Valid for integration tests verifying handler compliance
#    - Example: assert isinstance(handler, ProtocolDiscoveryOperations)
#
# WHY BOTH APPROACHES:
#   - Duck typing: Maximum flexibility, works with any implementation
#   - isinstance: Type safety, IDE support, explicit contract verification
#   - Tests may use EITHER or BOTH depending on context
#
# WHEN TO USE WHICH:
#   - Use duck typing helpers when testing generic interfaces
#   - Use isinstance when testing specific protocol implementations
#   - Use both when you want to verify BOTH interface shape AND type
#
# Related: src/omnibase_infra/protocols/ (protocol definitions with @runtime_checkable)
# =============================================================================


def assert_has_methods(
    obj: object, required_methods: list[str], *, protocol_name: str | None = None
) -> None:
    """Assert that an object has all required methods (duck typing conformance).

    This helper verifies protocol conformance via duck typing by checking
    for required method presence and callability. This approach works with
    any object regardless of explicit Protocol inheritance.

    For @runtime_checkable protocols, you can ALSO use isinstance() checks
    in addition to or instead of these helpers. Both approaches are valid
    in ONEX tests.

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
        ...     protocol_name="RegistryPolicy",
        ... )
    """
    name: str = protocol_name or obj.__class__.__name__
    for method_name in required_methods:
        assert hasattr(obj, method_name), f"{name} must have '{method_name}' method"
        # __len__ and __iter__ are special - they are callable via len()/iter()
        if not method_name.startswith("__"):
            assert callable(getattr(obj, method_name)), (
                f"{name}.{method_name} must be callable"
            )


def assert_has_async_methods(
    obj: object, required_methods: list[str], *, protocol_name: str | None = None
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
    name: str = protocol_name or obj.__class__.__name__
    for method_name in required_methods:
        assert hasattr(obj, method_name), f"{name} must have '{method_name}' method"
        method: object = getattr(obj, method_name)
        assert callable(method), f"{name}.{method_name} must be callable"
        assert asyncio.iscoroutinefunction(method), (
            f"{name}.{method_name} must be async (coroutine function)"
        )


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
    name: str = protocol_name or obj.__class__.__name__
    assert hasattr(obj, method_name), f"{name} must have '{method_name}' method"

    method: object = getattr(obj, method_name)
    sig: inspect.Signature = inspect.signature(method)
    params: list[str] = list(sig.parameters.keys())

    assert len(params) == len(expected_params), (
        f"{name}.{method_name} must have {len(expected_params)} parameters "
        f"({', '.join(expected_params)}), got {len(params)}: {params}"
    )

    for expected in expected_params:
        assert expected in params, (
            f"{name}.{method_name} must have '{expected}' parameter, got: {params}"
        )


# =============================================================================
# Registry-Specific Conformance Helpers
# =============================================================================


def assert_policy_registry_interface(registry: object) -> None:
    """Assert that an object implements the RegistryPolicy interface.

    Per ONEX conventions, protocol conformance is verified via duck typing.
    Collection-like protocols must include __len__ for complete duck typing.

    Args:
        registry: The object to verify as a RegistryPolicy implementation.

    Raises:
        AssertionError: If required methods are missing.

    Example:
        >>> async def example_test(container):
        ...     from omnibase_infra.runtime.registry_policy import RegistryPolicy
        ...     registry = await container.service_registry.resolve_service(RegistryPolicy)
        ...     assert_policy_registry_interface(registry)
        ...     assert len(registry) == 0  # Empty initially
    """
    required_methods = [
        "register",
        "register_policy",
        "get",
        "list_keys",
        "is_registered",
        "__len__",
    ]
    assert_has_methods(registry, required_methods, protocol_name="RegistryPolicy")


def assert_handler_registry_interface(registry: object) -> None:
    """Assert that an object implements the RegistryProtocolBinding interface.

    Per ONEX conventions, protocol conformance is verified via duck typing.
    Collection-like protocols must include __len__ for complete duck typing.

    Args:
        registry: The object to verify as a RegistryProtocolBinding implementation.

    Raises:
        AssertionError: If required methods are missing.

    Example:
        >>> async def example_test(container):
        ...     from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
        ...     registry = await container.service_registry.resolve_service(RegistryProtocolBinding)
        ...     assert_handler_registry_interface(registry)
        ...     assert len(registry) == 0
    """
    required_methods = [
        "register",
        "get",
        "list_protocols",
        "is_registered",
        "__len__",
    ]
    assert_has_methods(
        registry, required_methods, protocol_name="RegistryProtocolBinding"
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
        assert hasattr(dispatcher, prop), (
            f"ProtocolMessageDispatcher must have '{prop}' property"
        )

    assert hasattr(dispatcher, "handle"), (
        "ProtocolMessageDispatcher must have 'handle' method"
    )
    assert callable(dispatcher.handle), (
        "ProtocolMessageDispatcher.handle must be callable"
    )


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
        ...     # Configure the mock to return your service
        ...     mock_container.service_registry.resolve_service.return_value = some_service
        ...     # Call with await (resolve_service is async in omnibase_core 0.4.x+)
        ...     result = await mock_container.service_registry.resolve_service(SomeType)
        ...     assert result is some_service
    """
    container = MagicMock()

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
def simple_mock_container() -> MagicMock:
    """Create a simple mock ONEX container for basic node tests.

    This provides a minimal mock container with just the basic
    container.config attribute needed for NodeOrchestrator initialization.
    Use this for unit tests that don't need full container wiring.

    For tests requiring service_registry or async methods, use mock_container.
    For integration tests requiring real container behavior, use
    container_with_registries.

    Returns:
        MagicMock configured with minimal container.config attribute.

    Example::

        def test_orchestrator_creates(simple_mock_container: MagicMock) -> None:
            orchestrator = NodeRegistrationOrchestrator(simple_mock_container)
            assert orchestrator is not None

    """
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def container_with_policy_registry(mock_container: MagicMock) -> RegistryPolicy:
    """Create RegistryPolicy and configure mock container to resolve it.

    Provides a real RegistryPolicy instance registered in a mock container.
    This fixture demonstrates the container-based DI pattern for testing.

    Args:
        mock_container: Mock container fixture (automatically injected).

    Returns:
        RegistryPolicy instance that can be resolved from mock_container.

    Example:
        >>> async def test_container_based_policy_access(container_with_policy_registry, mock_container):
        ...     # Registry is already registered in mock_container
        ...     from omnibase_infra.runtime.registry_policy import RegistryPolicy
        ...     registry = await mock_container.service_registry.resolve_service(RegistryPolicy)
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
    from omnibase_infra.runtime.registry_policy import RegistryPolicy

    # Create real RegistryPolicy instance
    registry = RegistryPolicy()

    # Configure mock container to return this registry when resolved
    async def resolve_service_side_effect(interface_type: type) -> RegistryPolicy:
        if interface_type is RegistryPolicy:
            return registry
        raise ValueError(f"Service not registered: {interface_type}")

    mock_container.service_registry.resolve_service.side_effect = (
        resolve_service_side_effect
    )

    return registry


@pytest.fixture
async def container_with_registries() -> ModelONEXContainer:
    """Create real ONEX container with wired infrastructure services.

    Provides a fully wired ModelONEXContainer with RegistryPolicy and
    RegistryProtocolBinding registered as global services. This fixture
    demonstrates the real container-based DI pattern for integration tests.

    Note: This fixture is async because wire_infrastructure_services() is async.

    Important (OMN-1257):
        In omnibase_core 0.6.2+, container.service_registry may return None if:
        - enable_service_registry=False was passed to constructor
        - The ServiceRegistry module is not installed/available
        This fixture explicitly enables service_registry and validates it.

    Returns:
        ModelONEXContainer instance with infrastructure services wired.

    Raises:
        pytest.skip: If service_registry is None (ServiceRegistry module unavailable).

    Example:
        >>> async def test_with_real_container(container_with_registries):
        ...     from omnibase_infra.runtime.registry_policy import RegistryPolicy
        ...     from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
        ...
        ...     # Resolve services from real container (async)
        ...     policy_reg = await container_with_registries.service_registry.resolve_service(RegistryPolicy)
        ...     handler_reg = await container_with_registries.service_registry.resolve_service(RegistryProtocolBinding)
        ...
        ...     # Verify interface via duck typing (ONEX convention)
        ...     # Per ONEX conventions, check for required methods rather than isinstance
        ...     assert hasattr(policy_reg, "register_policy")
        ...     assert hasattr(handler_reg, "register")

    Raises:
        pytest.skip: If omnibase_core has a circular import bug causing
            service_registry to be None. This is a known issue in
            omnibase_core 0.6.2 where the import chain
            model_onex_container.py -> container_service_registry.py ->
            container/__init__.py -> container_service_resolver.py ->
            ModelONEXContainer (still loading) causes a circular import failure.
    """
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.util_container_wiring import (
        ServiceRegistryUnavailableError,
        wire_infrastructure_services,
    )

    # Create real container with service_registry explicitly enabled
    # In omnibase_core 0.6.2+, this may still return None if module unavailable
    container = ModelONEXContainer(enable_service_registry=True)

    # Check for omnibase_core circular import bug (service_registry is None)
    # This occurs in omnibase_core 0.6.2 due to circular import:
    # model_onex_container.py -> container_service_registry.py ->
    # container/__init__.py -> container_service_resolver.py ->
    # ModelONEXContainer (still loading) -> CIRCULAR IMPORT FAILURE
    if container.service_registry is None:
        pytest.skip(
            "Skipped: omnibase_core circular import bug - service_registry is None. "
            "This is a known issue in omnibase_core 0.6.2 where ServiceRegistry "
            "import fails during ModelONEXContainer initialization due to circular "
            "imports. See: model_onex_container.py -> container_service_registry.py "
            "-> container/__init__.py -> container_service_resolver.py -> "
            "ModelONEXContainer (still loading)"
        )

    # Additional validation: check that service_registry has required methods
    if not hasattr(container.service_registry, "register_instance"):
        pytest.skip(
            "Skipped: omnibase_core API incompatibility - service_registry missing "
            "'register_instance' method. This may indicate an omnibase_core version "
            "mismatch or incomplete ServiceRegistry initialization."
        )

    try:
        # Wire infrastructure services (async operation)
        await wire_infrastructure_services(container)
    except ServiceRegistryUnavailableError as e:
        pytest.skip(f"ServiceRegistry unavailable: {e}")

    # Return container. Note: ModelONEXContainer doesn't have explicit cleanup
    # methods currently. If future cleanup needs arise, change this to yield.
    return container


@pytest.fixture
async def container_with_handler_registry(
    container_with_registries: ModelONEXContainer,
) -> RegistryProtocolBinding:
    """Get RegistryProtocolBinding from wired container.

    Convenience fixture that extracts RegistryProtocolBinding from the
    container_with_registries fixture. Use this when you only need the
    handler registry without the full container.

    Note: This fixture is async because resolve_service() is async.

    Args:
        container_with_registries: Container fixture (automatically injected).

    Returns:
        RegistryProtocolBinding instance from container.

    Example:
        >>> async def test_handler_registry(container_with_handler_registry):
        ...     from omnibase_infra.runtime.handler_registry import HANDLER_TYPE_HTTP
        ...     container_with_handler_registry.register(HANDLER_TYPE_HTTP, MockHandler)
        ...     assert container_with_handler_registry.is_registered(HANDLER_TYPE_HTTP)
    """
    from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding

    registry: RegistryProtocolBinding = (
        await container_with_registries.service_registry.resolve_service(
            RegistryProtocolBinding
        )
    )
    return registry


# =============================================================================
# Infrastructure Cleanup Fixtures
# =============================================================================
# These fixtures ensure test isolation by cleaning up shared infrastructure
# resources (Consul, PostgreSQL, Kafka) after tests complete. They are designed
# to be used in integration tests that interact with real infrastructure.
#
# Related: tests/integration/registration/e2e/conftest.py (E2E-specific cleanup)
# =============================================================================


@pytest.fixture
async def cleanup_consul_test_services(
    mock_container: MagicMock,
) -> AsyncGenerator[None, None]:
    """Clean up orphaned Consul service registrations after each test.

    This fixture provides comprehensive Consul cleanup by:
    1. Yielding to let the test run
    2. After the test, querying all registered services
    3. Deregistering any services matching test patterns

    Args:
        mock_container: ONEX container mock for dependency injection.

    Test Service Identification Patterns:
        - Service ID starts with "test-"
        - Service ID contains "-test-" (e.g., "e2e-test-node-123")
        - Service name starts with "test"
        - Service name contains "integration-test"

    Usage:
        For tests that register Consul services and need cleanup,
        include this fixture. It handles cleanup even if the test fails.

        >>> async def test_consul_registration(cleanup_consul_test_services):
        ...     # Register a test service
        ...     await consul_client.register_service(
        ...         service_id="test-my-service-123",
        ...         service_name="test-service",
        ...         tags=["test"],
        ...     )
        ...     # Fixture will deregister after test completes

    Note:
        This fixture requires Consul to be available. It skips cleanup
        gracefully if Consul is not reachable or not configured.
    """
    import socket

    yield  # Let the test run

    # Check if Consul is configured and reachable
    consul_host = os.getenv("CONSUL_HOST")
    consul_port = int(os.getenv("CONSUL_PORT", "8500"))

    if not consul_host:
        return  # Consul not configured, skip cleanup

    # Quick reachability check
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            if sock.connect_ex((consul_host, consul_port)) != 0:
                return  # Consul not reachable
    except (OSError, TimeoutError):
        return  # Consul not reachable

    # Import and create handler for cleanup
    try:
        from omnibase_infra.handlers import HandlerConsul

        handler = HandlerConsul(mock_container)
        await handler.initialize(
            {
                "host": consul_host,
                "port": consul_port,
                "scheme": os.getenv("CONSUL_SCHEME", "http"),
                "timeout_seconds": 10.0,
            }
        )

        try:
            # Get all registered services
            # NOTE: consul.list_services is not yet implemented in HandlerConsul.
            # When implemented, it should return ModelHandlerOutput with services data.
            # For now, this will raise RuntimeHostError for unsupported operation,
            # which is caught by the outer exception handler.
            list_envelope = {
                "operation": "consul.list_services",
                "payload": {},
            }
            result = await handler.execute(list_envelope)

            # Access model attributes instead of dict keys
            # result is ModelHandlerOutput[ModelConsulHandlerResponse]
            if result and result.result and result.result.status == "success":
                # Access payload data via model attributes
                payload_data = result.result.payload.data
                # Services data structure depends on list_services implementation
                # Expected: dict mapping service_id -> service_info
                services = (
                    getattr(payload_data, "services", {})
                    if hasattr(payload_data, "services")
                    else {}
                )

                # Identify test services to cleanup
                test_service_ids: list[str] = []
                for service_id, service_info in services.items():
                    service_name = ""
                    # Access model attributes instead of dict keys
                    if hasattr(service_info, "name"):
                        service_name = service_info.name
                    elif hasattr(service_info, "Service"):
                        service_name = service_info.Service
                    elif isinstance(service_info, dict):
                        service_name = service_info.get("Service", "")
                    elif isinstance(service_info, list) and service_info:
                        service_name = str(service_info[0]) if service_info else ""

                    # Check if this is a test service
                    service_id_lower = service_id.lower()
                    service_name_lower = service_name.lower()
                    is_test_service = (
                        service_id.startswith(("test-", "e2e-"))
                        or "-test-" in service_id_lower
                        or service_name_lower.startswith("test")
                        or "integration-test" in service_name_lower
                    )

                    if is_test_service:
                        test_service_ids.append(service_id)

                # Deregister test services
                for service_id in test_service_ids:
                    try:
                        deregister_envelope = {
                            "operation": "consul.deregister",
                            "payload": {"service_id": service_id},
                        }
                        await handler.execute(deregister_envelope)
                        logger.debug(
                            "Successfully deregistered Consul test service: %s",
                            service_id,
                        )
                    except Exception as e:
                        # Note: exc_info omitted to prevent potential info leakage
                        logger.warning(
                            "Cleanup failed for Consul service %s: %s",
                            service_id,
                            sanitize_error_message(e),
                        )

        finally:
            await handler.shutdown()

    except Exception as e:
        # Note: exc_info omitted for consistency with other cleanup handlers
        logger.warning("Consul test cleanup failed: %s", sanitize_error_message(e))


@pytest.fixture
async def cleanup_postgres_test_projections() -> AsyncGenerator[None, None]:
    """Clean up stale PostgreSQL projection rows after tests.

    This fixture provides comprehensive PostgreSQL cleanup by:
    1. Yielding to let the test run
    2. After the test, deleting projection rows matching test patterns

    Cleanup Targets:
        - registration_projections table: Rows with entity_id matching test patterns
        - Patterns: UUID entity_id values (cleaned up by test-specific fixtures)
        - Rows where node_id starts with test prefixes are cleaned

    Table Cleanup Patterns:
        - registration_projections: Test node registrations

    Usage:
        For tests that create projection rows and need cleanup:

        >>> async def test_projector(cleanup_postgres_test_projections, postgres_pool):
        ...     # Create test projection
        ...     await projector.upsert(node_id=test_node_id, ...)
        ...     # Fixture will cleanup test patterns after test

    Note:
        This fixture requires PostgreSQL to be available. It skips cleanup
        gracefully if the database is not reachable or tables don't exist.

    Warning:
        PRODUCTION DATABASE SAFETY: This fixture executes DELETE operations
        against the configured database. The cleanup query uses pattern matching
        (LIKE '%test%', '%integration%') to target only test data. However:

        - NEVER run tests against a production database
        - Always verify OMNIBASE_INFRA_DB_URL points to a test/dev environment
        - The .env file should specify isolated test infrastructure
        - Production databases should use network isolation or read-only users

        The query intentionally uses restrictive WHERE clauses to minimize
        risk of accidental production data deletion.
    """

    yield  # Let the test run

    # Check if PostgreSQL is configured via OMNIBASE_INFRA_DB_URL or fallback vars
    from tests.helpers.util_postgres import PostgresConfig

    pg_config = PostgresConfig.from_env()
    if not pg_config.is_configured:
        return  # PostgreSQL not configured, skip cleanup
    postgres_dsn = pg_config.build_dsn()

    try:
        import asyncpg

        conn = await asyncpg.connect(postgres_dsn, timeout=10.0)

        try:
            # Clean up registration_projections with test-like metadata
            # This targets rows that may have been left by failed tests
            # by checking for common test patterns in metadata or status fields
            await conn.execute(
                """
                DELETE FROM registration_projections
                WHERE metadata::text LIKE '%test%'
                   OR metadata::text LIKE '%integration%'
                   OR status = 'TEST'
                """
            )
        except asyncpg.UndefinedTableError:
            pass  # Table doesn't exist, nothing to cleanup
        except Exception as e:
            # Note: exc_info omitted to prevent credential exposure in tracebacks
            # Exception is sanitized to prevent DSN/credential leakage
            logger.warning(
                "PostgreSQL projection cleanup query failed: %s",
                sanitize_error_message(e),
            )

        finally:
            await conn.close()

    except Exception as e:
        # Note: exc_info omitted to prevent credential exposure in tracebacks
        # (DSN contains password and would be visible in exception traceback)
        # Exception is sanitized to prevent DSN/credential leakage
        logger.warning("PostgreSQL test cleanup failed: %s", sanitize_error_message(e))


@pytest.fixture
async def cleanup_kafka_test_consumer_groups() -> AsyncGenerator[None, None]:
    """Reset Kafka consumer group offsets for test consumer groups after tests.

    This fixture provides Kafka consumer group cleanup by:
    1. Yielding to let the test run
    2. After the test, deleting consumer groups matching test patterns

    Test Consumer Group Identification Patterns:
        - Group ID starts with "test-"
        - Group ID contains "-test-"
        - Group ID starts with "e2e-"
        - Group ID contains "integration"

    Usage:
        For tests that create Kafka consumer groups and need cleanup:

        >>> async def test_kafka_consumer(cleanup_kafka_test_consumer_groups):
        ...     # Subscribe with test consumer group
        ...     await bus.subscribe("topic", "test-group-123", handler)
        ...     # Fixture will delete consumer group after test

    Note:
        This fixture requires Kafka to be available. It skips cleanup
        gracefully if Kafka is not reachable or not configured.
        Consumer groups are deleted using the Kafka admin client.
    """

    yield  # Let the test run

    # Check if Kafka is configured
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        return  # Kafka not configured, skip cleanup

    try:
        from aiokafka.admin import AIOKafkaAdminClient
        from aiokafka.errors import KafkaError

        admin_client = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)

        try:
            await admin_client.start()

            # List all consumer groups
            consumer_groups = await admin_client.list_consumer_groups()

            # Identify test consumer groups
            test_groups: list[str] = []
            for group_info in consumer_groups:
                group_id = (
                    group_info[0] if isinstance(group_info, tuple) else str(group_info)
                )

                group_id_lower = group_id.lower()
                is_test_group = (
                    group_id.startswith(("test-", "e2e-"))
                    or "-test-" in group_id_lower
                    or "integration" in group_id_lower
                )

                if is_test_group:
                    test_groups.append(group_id)

            # Delete test consumer groups
            if test_groups:
                try:
                    await admin_client.delete_consumer_groups(test_groups)
                except KafkaError as e:
                    # Note: exc_info omitted for consistency with other cleanup handlers
                    logger.warning(
                        "Kafka consumer group cleanup failed: %s",
                        sanitize_error_message(e),
                    )

        finally:
            await admin_client.close()

    except Exception as e:
        # Note: exc_info omitted for consistency with other cleanup handlers
        logger.warning("Kafka test cleanup failed: %s", sanitize_error_message(e))


@pytest.fixture
async def full_infrastructure_cleanup(
    cleanup_consul_test_services: None,
    cleanup_postgres_test_projections: None,
    cleanup_kafka_test_consumer_groups: None,
) -> None:
    """Combined fixture that provides cleanup for all infrastructure components.

    This is a convenience fixture that combines all infrastructure cleanup
    fixtures into a single dependency. Use this for E2E tests that interact
    with multiple infrastructure components.

    Components Cleaned:
        - Consul: Test service registrations (services matching test patterns)
        - PostgreSQL: Test projection rows (rows matching test patterns)
        - Kafka: Test consumer groups (groups matching test patterns)

    Usage:
        >>> async def test_full_e2e_flow(full_infrastructure_cleanup):
        ...     # Test that uses Consul, PostgreSQL, and Kafka
        ...     # All test artifacts will be cleaned up after test

    Note:
        Each cleanup fixture operates independently and handles errors
        gracefully. If one infrastructure component is unavailable,
        cleanup for other components will still proceed.

        The dependent fixtures (cleanup_consul_test_services, etc.) use
        yield and handle their own teardown, so this fixture returns
        immediately after they yield.
    """
    return  # Dependent fixtures handle their own teardown


# =============================================================================
# Dependency Materialization Skip Fixture
# =============================================================================
# Moved to tests/unit/conftest.py to scope to unit tests only.
# Integration tests that need this mock should define their own local fixture.
# See tests/unit/conftest.py for the implementation.
# =============================================================================


# =============================================================================
# RuntimeHostProcess Handler Seeding Helpers
# =============================================================================
# Canonical implementations live in ``tests.helpers.runtime_helpers``.
# Re-exported here for backwards compatibility so that existing imports of
# ``from tests.conftest import make_runtime_config, seed_mock_handlers``
# continue to work.  New code should import directly from
# ``tests.helpers.runtime_helpers``.
# =============================================================================

from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers


@pytest.fixture
def mock_runtime_handler() -> MagicMock:
    """Create a pre-configured mock handler suitable for runtime handler seeding.

    Returns a MagicMock configured with:
    - execute: AsyncMock for handling envelopes
    - initialize: AsyncMock for handler initialization
    - shutdown: AsyncMock for cleanup
    - health_check: AsyncMock returning {"healthy": True}
    - initialized: True (for health check compatibility)

    This fixture is useful when tests need access to the mock handler
    for assertions or additional configuration.

    Returns:
        MagicMock configured as a minimal handler implementation.

    Example:
        >>> async def test_something(mock_runtime_handler):
        ...     process = RuntimeHostProcess()
        ...     process._handlers = {"mock": mock_runtime_handler}
        ...     await process.start()
        ...     mock_runtime_handler.health_check.assert_called()
    """
    mock_handler = MagicMock()
    mock_handler.execute = AsyncMock(return_value={"success": True, "result": "mock"})
    mock_handler.initialize = AsyncMock()
    mock_handler.shutdown = AsyncMock()
    mock_handler.health_check = AsyncMock(return_value={"healthy": True})
    mock_handler.initialized = True
    return mock_handler


# =============================================================================
# Event Bus Fixtures
# =============================================================================
# These fixtures provide in-memory event bus instances for testing.
# The EventBusInmemory uses async yield pattern to guarantee cleanup
# even when tests fail, preventing resource leaks.
#
# Configuration Defaults:
#   - environment: "local" (appropriate for local development/testing)
#   - group: "default" (generic consumer group)
#   These defaults align with local development scenarios. Tests can
#   override by creating their own fixtures with custom values.
#
# Transport Type:
#   EventBusInmemory uses EnumInfraTransportType.INMEMORY (not KAFKA)
#   to correctly identify its transport in error contexts and logging.
# =============================================================================


@pytest.fixture
async def event_bus() -> AsyncGenerator[object, None]:
    """Create and start an in-memory event bus with guaranteed cleanup.

    This fixture provides an EventBusInmemory instance configured for testing.
    The async yield pattern ensures proper cleanup even if tests fail.

    Default Configuration:
        - environment: "test" (identifies test environment in logs)
        - group: "test-group" (test-specific consumer group)
        - max_history: 1000 (sufficient for most test scenarios)

    Yields:
        Started EventBusInmemory instance.

    Note:
        The cleanup (await bus.close()) is guaranteed to run after each test,
        even if the test fails. This prevents resource leaks in test suites.

    Example:
        >>> async def test_publish_subscribe(event_bus):
        ...     received = []
        ...     async def handler(msg):
        ...         received.append(msg)
        ...     await event_bus.subscribe("topic", "group", handler)
        ...     await event_bus.publish("topic", None, b"test")
        ...     assert len(received) == 1
    """
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

    bus = EventBusInmemory(environment="test", group="test-group", max_history=1000)
    await bus.start()
    yield bus
    await bus.close()

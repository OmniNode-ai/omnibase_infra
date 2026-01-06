"""Pytest configuration and shared fixtures for omnibase_infra tests."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.utils import sanitize_error_message

# Module-level logger for test cleanup diagnostics
logger = logging.getLogger(__name__)

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
            assert callable(getattr(obj, method_name)), (
                f"{name}.{method_name} must be callable"
            )


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
        assert expected in params, (
            f"{name}.{method_name} must have '{expected}' parameter, got: {params}"
        )


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
        ...     # Mock container is ready to use (async methods)
        ...     mock_container.service_registry.resolve_service.return_value = some_service
    """
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

    Important (OMN-1257):
        In omnibase_core 0.6.2+, container.service_registry may return None if:
        - enable_service_registry=False was passed to constructor
        - The ServiceRegistry module is not installed/available
        This fixture explicitly enables service_registry and validates it.

    Yields:
        ModelONEXContainer instance with infrastructure services wired.

    Raises:
        pytest.skip: If service_registry is None (ServiceRegistry module unavailable).

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

    from omnibase_infra.runtime.container_wiring import (
        ServiceRegistryUnavailableError,
        wire_infrastructure_services,
    )

    # Create real container with service_registry explicitly enabled
    # In omnibase_core 0.6.2+, this may still return None if module unavailable
    container = ModelONEXContainer(enable_service_registry=True)

    # Check if service_registry is available (OMN-1257)
    if container.service_registry is None:
        pytest.skip(
            "ServiceRegistry not available in omnibase_core. "
            "Tests requiring container.service_registry will be skipped. "
            "Check omnibase_core logs for 'ServiceRegistry not available' warnings."
        )

    try:
        # Wire infrastructure services (async operation)
        await wire_infrastructure_services(container)
    except ServiceRegistryUnavailableError as e:
        pytest.skip(f"ServiceRegistry unavailable: {e}")

    # Yield container for proper fixture teardown semantics.
    # ModelONEXContainer doesn't have explicit cleanup methods currently,
    # but using yield allows for future cleanup needs and ensures proper
    # pytest async fixture lifecycle management.
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
async def cleanup_consul_test_services():
    """Clean up orphaned Consul service registrations after each test.

    This fixture provides comprehensive Consul cleanup by:
    1. Yielding to let the test run
    2. After the test, querying all registered services
    3. Deregistering any services matching test patterns

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
    import os
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
        from omnibase_infra.handlers import ConsulHandler

        handler = ConsulHandler()
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
            # NOTE: consul.list_services is not yet implemented in ConsulHandler.
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
        logger.warning(
            "Consul test cleanup failed: %s",
            sanitize_error_message(e),
        )


@pytest.fixture
async def cleanup_postgres_test_projections():
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
        - Always verify POSTGRES_HOST points to a test/dev environment
        - The .env file should specify isolated test infrastructure
        - Production databases should use network isolation or read-only users

        The query intentionally uses restrictive WHERE clauses to minimize
        risk of accidental production data deletion.
    """
    import os

    yield  # Let the test run

    # Check if PostgreSQL is configured
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    if not postgres_host or not postgres_password:
        return  # PostgreSQL not configured, skip cleanup

    # Build connection string
    postgres_port = os.getenv("POSTGRES_PORT", "5436")
    postgres_database = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")

    dsn = (
        f"postgresql://{postgres_user}:{postgres_password}"
        f"@{postgres_host}:{postgres_port}/{postgres_database}"
    )

    try:
        import asyncpg

        conn = await asyncpg.connect(dsn, timeout=10.0)

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
                """,
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
        logger.warning(
            "PostgreSQL test cleanup failed: %s",
            sanitize_error_message(e),
        )


@pytest.fixture
async def cleanup_kafka_test_consumer_groups():
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
    import os

    yield  # Let the test run

    # Check if Kafka is configured
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        return  # Kafka not configured, skip cleanup

    try:
        from aiokafka.admin import AIOKafkaAdminClient
        from aiokafka.errors import KafkaError

        admin_client = AIOKafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
        )

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
        logger.warning(
            "Kafka test cleanup failed: %s",
            sanitize_error_message(e),
        )


@pytest.fixture
async def full_infrastructure_cleanup(
    cleanup_consul_test_services,
    cleanup_postgres_test_projections,
    cleanup_kafka_test_consumer_groups,
):
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

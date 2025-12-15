"""Unit tests for handler_registry module.

Tests follow TDD approach:
1. Write tests first (red phase)
2. Implement registry classes (green phase)
3. Refactor if needed (refactor phase)

All tests validate:
- Handler registration and retrieval
- Event bus registration and retrieval
- Singleton pattern implementation
- Thread safety
- Error handling for missing registrations
- Convenience functions
- Operation prefix routing validation (OMN-807)
"""

from __future__ import annotations

import threading

import pytest

from omnibase_infra.errors import (
    ModelInfraErrorContext,
    RuntimeHostError,
    UnknownHandlerTypeError,
)
from omnibase_infra.runtime import handler_registry as registry_module
from omnibase_infra.runtime.handler_registry import (
    EVENT_BUS_INMEMORY,
    EVENT_BUS_KAFKA,
    HANDLER_TYPE_CONSUL,
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_GRPC,
    HANDLER_TYPE_HTTP,
    HANDLER_TYPE_KAFKA,
    HANDLER_TYPE_VALKEY,
    HANDLER_TYPE_VAULT,
    EventBusBindingRegistry,
    ProtocolBindingRegistry,
    RegistryError,
    get_event_bus_class,
    get_event_bus_registry,
    get_handler_class,
    get_handler_registry,
    register_handlers_from_config,
)
from omnibase_infra.runtime.models import ModelProtocolRegistrationConfig

# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockHttpHandler:
    """Mock HTTP handler for testing."""


class MockDbHandler:
    """Mock database handler for testing."""


class MockKafkaHandler:
    """Mock Kafka handler for testing."""


class MockVaultHandler:
    """Mock Vault handler for testing."""


class MockConsulHandler:
    """Mock Consul handler for testing."""


class MockValkeyHandler:
    """Mock Valkey handler for testing."""


class MockGrpcHandler:
    """Mock gRPC handler for testing."""


class MockInMemoryEventBus:
    """Mock in-memory event bus for testing."""


class MockKafkaEventBus:
    """Mock Kafka event bus for testing."""


class MockAlternativeEventBus:
    """Alternative mock event bus for testing."""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def handler_registry() -> ProtocolBindingRegistry:
    """Provide a fresh ProtocolBindingRegistry instance for each test."""
    return ProtocolBindingRegistry()


@pytest.fixture
def event_bus_registry() -> EventBusBindingRegistry:
    """Provide a fresh EventBusBindingRegistry instance for each test."""
    return EventBusBindingRegistry()


@pytest.fixture
def populated_handler_registry() -> ProtocolBindingRegistry:
    """Provide a ProtocolBindingRegistry with pre-registered handlers."""
    registry = ProtocolBindingRegistry()
    registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
    registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)
    registry.register(HANDLER_TYPE_KAFKA, MockKafkaHandler)
    return registry


@pytest.fixture
def populated_event_bus_registry() -> EventBusBindingRegistry:
    """Provide an EventBusBindingRegistry with pre-registered buses."""
    registry = EventBusBindingRegistry()
    registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
    return registry


@pytest.fixture(autouse=True)
def reset_singletons() -> None:  # type: ignore[misc]
    """Reset singleton instances before each test.

    This ensures tests are isolated and don't affect each other
    through the singleton state.
    """
    with registry_module._singleton_lock:
        registry_module._handler_registry = None
        registry_module._event_bus_registry = None
    yield
    # Also reset after test
    with registry_module._singleton_lock:
        registry_module._handler_registry = None
        registry_module._event_bus_registry = None


# =============================================================================
# ProtocolBindingRegistry Tests
# =============================================================================


class TestHandlerRegistryBasics:
    """Basic tests for ProtocolBindingRegistry class."""

    def test_empty_registry_initialization(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that a new registry is empty."""
        assert len(handler_registry) == 0
        assert handler_registry.list_protocols() == []

    def test_register_handler_success(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test registering a handler successfully."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        assert handler_registry.is_registered(HANDLER_TYPE_HTTP)
        assert len(handler_registry) == 1

    def test_register_multiple_handlers(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test registering multiple handlers."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)
        handler_registry.register(HANDLER_TYPE_KAFKA, MockKafkaHandler)
        assert len(handler_registry) == 3
        assert handler_registry.is_registered(HANDLER_TYPE_HTTP)
        assert handler_registry.is_registered(HANDLER_TYPE_DATABASE)
        assert handler_registry.is_registered(HANDLER_TYPE_KAFKA)

    def test_get_registered_handler(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test retrieving a registered handler class."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_cls = handler_registry.get(HANDLER_TYPE_HTTP)
        assert handler_cls is MockHttpHandler

    def test_get_unregistered_handler_raises(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that getting an unregistered handler raises RegistryError."""
        with pytest.raises(RegistryError) as exc_info:
            handler_registry.get("unknown_protocol")
        assert "unknown_protocol" in str(exc_info.value)
        assert "No handler registered" in str(exc_info.value)

    def test_get_unregistered_handler_error_contains_registered_list(
        self, populated_handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that RegistryError message includes list of registered protocols."""
        with pytest.raises(RegistryError) as exc_info:
            populated_handler_registry.get("unknown_protocol")
        error_msg = str(exc_info.value)
        # Should list registered protocols
        assert "database" in error_msg or "http" in error_msg or "kafka" in error_msg


class TestHandlerRegistryListProtocols:
    """Tests for list_protocols method."""

    def test_list_protocols_empty(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test list_protocols returns empty list for empty registry."""
        assert handler_registry.list_protocols() == []

    def test_list_protocols_with_handlers(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test list_protocols returns registered protocol types."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)
        protocols = handler_registry.list_protocols()
        assert HANDLER_TYPE_HTTP in protocols
        assert HANDLER_TYPE_DATABASE in protocols

    def test_list_protocols_sorted_alphabetically(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test list_protocols returns protocols in alphabetical order."""
        # Register in non-alphabetical order
        handler_registry.register(HANDLER_TYPE_KAFKA, MockKafkaHandler)
        handler_registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        protocols = handler_registry.list_protocols()
        # Should be sorted
        assert protocols == sorted(protocols)


class TestHandlerRegistryIsRegistered:
    """Tests for is_registered method."""

    def test_is_registered_true(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test is_registered returns True for registered handlers."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        assert handler_registry.is_registered(HANDLER_TYPE_HTTP) is True

    def test_is_registered_false(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test is_registered returns False for unregistered handlers."""
        assert handler_registry.is_registered(HANDLER_TYPE_HTTP) is False

    def test_is_registered_after_unregister(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test is_registered returns False after handler is unregistered."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_registry.unregister(HANDLER_TYPE_HTTP)
        assert handler_registry.is_registered(HANDLER_TYPE_HTTP) is False


class TestHandlerRegistryContains:
    """Tests for __contains__ dunder method (in operator)."""

    def test_contains_registered_handler(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test 'in' operator works for registered handlers."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        assert HANDLER_TYPE_HTTP in handler_registry

    def test_not_contains_unregistered_handler(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test 'in' operator returns False for unregistered handlers."""
        assert HANDLER_TYPE_HTTP not in handler_registry

    def test_contains_multiple_handlers(
        self, populated_handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test 'in' operator works correctly with multiple handlers."""
        assert HANDLER_TYPE_HTTP in populated_handler_registry
        assert HANDLER_TYPE_DATABASE in populated_handler_registry
        assert HANDLER_TYPE_KAFKA in populated_handler_registry
        assert HANDLER_TYPE_VAULT not in populated_handler_registry


class TestHandlerRegistryOverwrite:
    """Tests for handler overwrite behavior."""

    def test_register_duplicate_overwrites(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that registering same protocol type overwrites existing handler."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_registry.register(HANDLER_TYPE_HTTP, MockDbHandler)
        # Should now return the new handler
        handler_cls = handler_registry.get(HANDLER_TYPE_HTTP)
        assert handler_cls is MockDbHandler
        # Count should remain 1
        assert len(handler_registry) == 1

    def test_overwrite_preserves_other_registrations(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that overwriting one handler doesn't affect others."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)
        handler_registry.register(HANDLER_TYPE_HTTP, MockKafkaHandler)  # Overwrite

        assert handler_registry.get(HANDLER_TYPE_HTTP) is MockKafkaHandler
        assert handler_registry.get(HANDLER_TYPE_DATABASE) is MockDbHandler


class TestHandlerRegistryUnregister:
    """Tests for unregister method."""

    def test_unregister_existing_handler(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test unregistering an existing handler returns True."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        result = handler_registry.unregister(HANDLER_TYPE_HTTP)
        assert result is True
        assert HANDLER_TYPE_HTTP not in handler_registry

    def test_unregister_nonexistent_handler(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test unregistering a non-existent handler returns False."""
        result = handler_registry.unregister(HANDLER_TYPE_HTTP)
        assert result is False

    def test_unregister_idempotent(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that unregistering twice is safe and returns False second time."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        assert handler_registry.unregister(HANDLER_TYPE_HTTP) is True
        assert handler_registry.unregister(HANDLER_TYPE_HTTP) is False

    def test_unregister_preserves_other_registrations(
        self, populated_handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that unregistering one handler doesn't affect others."""
        populated_handler_registry.unregister(HANDLER_TYPE_HTTP)
        assert HANDLER_TYPE_DATABASE in populated_handler_registry
        assert HANDLER_TYPE_KAFKA in populated_handler_registry


class TestHandlerRegistryClear:
    """Tests for clear method."""

    def test_clear_empty_registry(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test clearing an empty registry is safe."""
        handler_registry.clear()
        assert len(handler_registry) == 0

    def test_clear_populated_registry(
        self, populated_handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test clearing removes all registrations."""
        populated_handler_registry.clear()
        assert len(populated_handler_registry) == 0
        assert populated_handler_registry.list_protocols() == []
        assert HANDLER_TYPE_HTTP not in populated_handler_registry


class TestHandlerRegistryLen:
    """Tests for __len__ dunder method."""

    def test_len_empty_registry(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test len() returns 0 for empty registry."""
        assert len(handler_registry) == 0

    def test_len_after_registration(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test len() returns correct count after registrations."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        assert len(handler_registry) == 1
        handler_registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)
        assert len(handler_registry) == 2

    def test_len_after_unregister(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test len() decreases after unregistration."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)
        handler_registry.unregister(HANDLER_TYPE_HTTP)
        assert len(handler_registry) == 1

    def test_len_after_clear(
        self, populated_handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test len() is 0 after clear."""
        populated_handler_registry.clear()
        assert len(populated_handler_registry) == 0


class TestHandlerRegistryThreadSafety:
    """Tests for thread safety of ProtocolBindingRegistry."""

    def test_concurrent_registration(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that concurrent registrations are thread-safe."""
        handlers = [
            (HANDLER_TYPE_HTTP, MockHttpHandler),
            (HANDLER_TYPE_DATABASE, MockDbHandler),
            (HANDLER_TYPE_KAFKA, MockKafkaHandler),
            (HANDLER_TYPE_VAULT, MockVaultHandler),
            (HANDLER_TYPE_CONSUL, MockConsulHandler),
            (HANDLER_TYPE_VALKEY, MockValkeyHandler),
            (HANDLER_TYPE_GRPC, MockGrpcHandler),
        ]

        def register_handler(protocol: str, cls: type) -> None:
            handler_registry.register(protocol, cls)

        threads = [
            threading.Thread(target=register_handler, args=(proto, cls))
            for proto, cls in handlers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All handlers should be registered
        assert len(handler_registry) == len(handlers)
        for proto, cls in handlers:
            assert handler_registry.get(proto) is cls

    def test_concurrent_read_write(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test concurrent reads and writes don't cause data corruption."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)

        errors: list[Exception] = []

        def read_handler() -> None:
            try:
                for _ in range(100):
                    handler_registry.get(HANDLER_TYPE_HTTP)
                    handler_registry.is_registered(HANDLER_TYPE_HTTP)
                    handler_registry.list_protocols()
            except Exception as e:
                errors.append(e)

        def write_handler() -> None:
            try:
                for _ in range(100):
                    handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=read_handler) for _ in range(5)]
        writers = [threading.Thread(target=write_handler) for _ in range(2)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        assert len(errors) == 0, f"Errors occurred during concurrent access: {errors}"


# =============================================================================
# EventBusBindingRegistry Tests
# =============================================================================


class TestEventBusRegistryBasics:
    """Basic tests for EventBusBindingRegistry class."""

    def test_empty_registry_initialization(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test that a new registry is empty."""
        assert event_bus_registry.list_bus_kinds() == []

    def test_register_event_bus_success(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test registering an event bus successfully."""
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        assert event_bus_registry.is_registered(EVENT_BUS_INMEMORY)

    def test_get_registered_event_bus(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test retrieving a registered event bus class."""
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        bus_cls = event_bus_registry.get(EVENT_BUS_INMEMORY)
        assert bus_cls is MockInMemoryEventBus

    def test_get_unregistered_event_bus_raises(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test that getting an unregistered bus raises RuntimeHostError."""
        with pytest.raises(RuntimeHostError) as exc_info:
            event_bus_registry.get("unknown_bus")
        assert "unknown_bus" in str(exc_info.value)
        assert "not registered" in str(exc_info.value)


class TestEventBusRegistryListBusKinds:
    """Tests for list_bus_kinds method."""

    def test_list_bus_kinds_empty(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test list_bus_kinds returns empty list for empty registry."""
        assert event_bus_registry.list_bus_kinds() == []

    def test_list_bus_kinds_with_buses(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test list_bus_kinds returns registered bus kinds."""
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        event_bus_registry.register(EVENT_BUS_KAFKA, MockKafkaEventBus)
        kinds = event_bus_registry.list_bus_kinds()
        assert EVENT_BUS_INMEMORY in kinds
        assert EVENT_BUS_KAFKA in kinds


class TestEventBusRegistryIsRegistered:
    """Tests for is_registered method."""

    def test_is_registered_bus_true(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test is_registered returns True for registered bus."""
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        assert event_bus_registry.is_registered(EVENT_BUS_INMEMORY) is True

    def test_is_registered_bus_false(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test is_registered returns False for unregistered bus."""
        assert event_bus_registry.is_registered(EVENT_BUS_KAFKA) is False


class TestEventBusRegistryDuplicateRaises:
    """Tests for duplicate registration behavior."""

    def test_register_duplicate_raises_error(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test that registering same bus_kind twice raises RuntimeHostError."""
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        with pytest.raises(RuntimeHostError) as exc_info:
            event_bus_registry.register(EVENT_BUS_INMEMORY, MockAlternativeEventBus)
        assert EVENT_BUS_INMEMORY in str(exc_info.value)
        assert "already registered" in str(exc_info.value)

    def test_register_duplicate_preserves_original(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test that failed duplicate registration preserves original."""
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        with pytest.raises(RuntimeHostError):
            event_bus_registry.register(EVENT_BUS_INMEMORY, MockAlternativeEventBus)
        # Original should still be there
        bus_cls = event_bus_registry.get(EVENT_BUS_INMEMORY)
        assert bus_cls is MockInMemoryEventBus

    def test_register_duplicate_error_contains_existing_class_name(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test that duplicate error includes existing class name in context."""
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        with pytest.raises(RuntimeHostError) as exc_info:
            event_bus_registry.register(EVENT_BUS_INMEMORY, MockAlternativeEventBus)
        # Error should have existing_class in context
        error = exc_info.value
        assert error.model.context.get("existing_class") == "MockInMemoryEventBus"


class TestEventBusRegistryThreadSafety:
    """Tests for thread safety of EventBusBindingRegistry."""

    def test_concurrent_registration_different_kinds(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test concurrent registrations of different bus kinds."""
        buses = [
            (EVENT_BUS_INMEMORY, MockInMemoryEventBus),
            (EVENT_BUS_KAFKA, MockKafkaEventBus),
            ("custom1", MockAlternativeEventBus),
        ]

        def register_bus(kind: str, cls: type) -> None:
            event_bus_registry.register(kind, cls)

        threads = [
            threading.Thread(target=register_bus, args=(kind, cls))
            for kind, cls in buses
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All buses should be registered
        for kind, cls in buses:
            assert event_bus_registry.get(kind) is cls


# =============================================================================
# Singleton Tests
# =============================================================================


class TestHandlerRegistrySingleton:
    """Tests for handler registry singleton pattern."""

    def test_get_handler_registry_singleton(self) -> None:
        """Test that get_handler_registry returns same instance each call."""
        registry1 = get_handler_registry()
        registry2 = get_handler_registry()
        assert registry1 is registry2

    def test_singleton_is_handler_registry_instance(self) -> None:
        """Test that singleton is a ProtocolBindingRegistry instance."""
        registry = get_handler_registry()
        assert isinstance(registry, ProtocolBindingRegistry)

    def test_singleton_state_persists(self) -> None:
        """Test that modifications to singleton persist across calls."""
        registry1 = get_handler_registry()
        registry1.register(HANDLER_TYPE_HTTP, MockHttpHandler)

        registry2 = get_handler_registry()
        assert registry2.is_registered(HANDLER_TYPE_HTTP)
        assert registry2.get(HANDLER_TYPE_HTTP) is MockHttpHandler

    def test_singleton_thread_safe_initialization(self) -> None:
        """Test that singleton initialization is thread-safe."""
        registries: list[ProtocolBindingRegistry] = []

        def get_registry() -> None:
            registries.append(get_handler_registry())

        threads = [threading.Thread(target=get_registry) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert len(registries) == 10
        first = registries[0]
        for reg in registries:
            assert reg is first


class TestEventBusRegistrySingleton:
    """Tests for event bus registry singleton pattern."""

    def test_get_event_bus_registry_singleton(self) -> None:
        """Test that get_event_bus_registry returns same instance each call."""
        registry1 = get_event_bus_registry()
        registry2 = get_event_bus_registry()
        assert registry1 is registry2

    def test_singleton_is_event_bus_registry_instance(self) -> None:
        """Test that singleton is an EventBusBindingRegistry instance."""
        registry = get_event_bus_registry()
        assert isinstance(registry, EventBusBindingRegistry)

    def test_singleton_state_persists(self) -> None:
        """Test that modifications to singleton persist across calls."""
        registry1 = get_event_bus_registry()
        registry1.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)

        registry2 = get_event_bus_registry()
        assert registry2.is_registered(EVENT_BUS_INMEMORY)
        assert registry2.get(EVENT_BUS_INMEMORY) is MockInMemoryEventBus

    def test_singleton_thread_safe_initialization(self) -> None:
        """Test that singleton initialization is thread-safe."""
        registries: list[EventBusBindingRegistry] = []

        def get_registry() -> None:
            registries.append(get_event_bus_registry())

        threads = [threading.Thread(target=get_registry) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert len(registries) == 10
        first = registries[0]
        for reg in registries:
            assert reg is first


class TestSingletonIndependence:
    """Tests that handler and event bus singletons are independent."""

    def test_singletons_are_different_instances(self) -> None:
        """Test that handler and event bus singletons are different objects."""
        handler_reg = get_handler_registry()
        event_bus_reg = get_event_bus_registry()
        assert handler_reg is not event_bus_reg

    def test_singletons_are_different_types(self) -> None:
        """Test that singletons have correct types."""
        handler_reg = get_handler_registry()
        event_bus_reg = get_event_bus_registry()
        assert type(handler_reg) is ProtocolBindingRegistry
        assert type(event_bus_reg) is EventBusBindingRegistry


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestGetHandlerClass:
    """Tests for get_handler_class convenience function."""

    def test_get_handler_class_uses_singleton(self) -> None:
        """Test that get_handler_class uses singleton registry."""
        # Register via singleton
        get_handler_registry().register(HANDLER_TYPE_HTTP, MockHttpHandler)
        # Retrieve via convenience function
        handler_cls = get_handler_class(HANDLER_TYPE_HTTP)
        assert handler_cls is MockHttpHandler

    def test_get_handler_class_raises_for_unknown(self) -> None:
        """Test that get_handler_class raises RegistryError for unknown type."""
        with pytest.raises(RegistryError) as exc_info:
            get_handler_class("nonexistent")
        assert "nonexistent" in str(exc_info.value)


class TestGetEventBusClass:
    """Tests for get_event_bus_class convenience function."""

    def test_get_event_bus_class_uses_singleton(self) -> None:
        """Test that get_event_bus_class uses singleton registry."""
        # Register via singleton
        get_event_bus_registry().register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        # Retrieve via convenience function
        bus_cls = get_event_bus_class(EVENT_BUS_INMEMORY)
        assert bus_cls is MockInMemoryEventBus

    def test_get_event_bus_class_raises_for_unknown(self) -> None:
        """Test that get_event_bus_class raises RuntimeHostError for unknown kind."""
        with pytest.raises(RuntimeHostError) as exc_info:
            get_event_bus_class("nonexistent")
        assert "nonexistent" in str(exc_info.value)


# =============================================================================
# Register Handlers From Config Tests
# =============================================================================


class TestRegisterHandlersFromConfig:
    """Tests for register_handlers_from_config function."""

    def test_empty_config_does_not_raise(self) -> None:
        """Test that empty config list doesn't raise."""
        register_handlers_from_config(runtime=None, protocol_configs=[])
        # Should not raise

    def test_disabled_handlers_skipped(self) -> None:
        """Test that handlers with enabled=False are skipped."""
        configs = [
            ModelProtocolRegistrationConfig(
                type="http", protocol_class="HttpHandler", enabled=False
            ),
        ]
        # This should not raise even though the handler isn't actually registered
        register_handlers_from_config(runtime=None, protocol_configs=configs)

    def test_enabled_handlers_processed(self) -> None:
        """Test that handlers with enabled=True are processed."""
        configs = [
            ModelProtocolRegistrationConfig(
                type="http", protocol_class="HttpHandler", enabled=True
            ),
            ModelProtocolRegistrationConfig(
                type="db", protocol_class="PostgresHandler"
            ),  # Default enabled
        ]
        # This should not raise - placeholder implementation just validates structure
        register_handlers_from_config(runtime=None, protocol_configs=configs)

    def test_missing_type_or_class_skipped(self) -> None:
        """Test that configs missing type or class are safely handled."""
        configs = [
            ModelProtocolRegistrationConfig(type="http", enabled=True),  # Missing class
            ModelProtocolRegistrationConfig(type="db"),  # Missing class (None)
        ]
        # Should not raise
        register_handlers_from_config(runtime=None, protocol_configs=configs)


# =============================================================================
# RegistryError Tests
# =============================================================================


class TestRegistryError:
    """Tests for RegistryError exception class."""

    def test_registry_error_message(self) -> None:
        """Test RegistryError preserves message."""
        error = RegistryError("Test error message")
        assert "Test error message" in str(error)

    def test_registry_error_with_protocol_type(self) -> None:
        """Test RegistryError with protocol_type context."""
        error = RegistryError(
            "Handler not found",
            protocol_type="unknown_protocol",
        )
        # Error should have protocol_type in context
        assert "Handler not found" in str(error)

    def test_registry_error_with_extra_context(self) -> None:
        """Test RegistryError with extra context kwargs."""
        error = RegistryError(
            "Handler not found",
            protocol_type="http",
            registered_protocols=["db", "kafka"],
        )
        assert "Handler not found" in str(error)

    def test_registry_error_is_exception(self) -> None:
        """Test RegistryError is an Exception."""
        error = RegistryError("Test")
        assert isinstance(error, Exception)


# =============================================================================
# Handler Type Constants Tests
# =============================================================================


class TestHandlerTypeConstants:
    """Tests for handler type constants."""

    def test_handler_type_constants_are_strings(self) -> None:
        """Test that all handler type constants are strings."""
        constants = [
            HANDLER_TYPE_HTTP,
            HANDLER_TYPE_DATABASE,
            HANDLER_TYPE_KAFKA,
            HANDLER_TYPE_VAULT,
            HANDLER_TYPE_CONSUL,
            HANDLER_TYPE_VALKEY,
            HANDLER_TYPE_GRPC,
        ]
        for const in constants:
            assert isinstance(const, str)

    def test_handler_type_constants_unique(self) -> None:
        """Test that all handler type constants are unique."""
        constants = [
            HANDLER_TYPE_HTTP,
            HANDLER_TYPE_DATABASE,
            HANDLER_TYPE_KAFKA,
            HANDLER_TYPE_VAULT,
            HANDLER_TYPE_CONSUL,
            HANDLER_TYPE_VALKEY,
            HANDLER_TYPE_GRPC,
        ]
        assert len(constants) == len(set(constants))


class TestEventBusKindConstants:
    """Tests for event bus kind constants."""

    def test_event_bus_constants_are_strings(self) -> None:
        """Test that all event bus kind constants are strings."""
        constants = [
            EVENT_BUS_INMEMORY,
            EVENT_BUS_KAFKA,
        ]
        for const in constants:
            assert isinstance(const, str)

    def test_event_bus_constants_unique(self) -> None:
        """Test that all event bus kind constants are unique."""
        constants = [
            EVENT_BUS_INMEMORY,
            EVENT_BUS_KAFKA,
        ]
        assert len(constants) == len(set(constants))


# =============================================================================
# Integration Tests
# =============================================================================


class TestHandlerRegistryIntegration:
    """Integration tests for ProtocolBindingRegistry with real-world scenarios."""

    def test_full_registration_workflow(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test complete workflow: register, get, unregister, re-register."""
        # Register
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        assert handler_registry.get(HANDLER_TYPE_HTTP) is MockHttpHandler

        # Unregister
        handler_registry.unregister(HANDLER_TYPE_HTTP)
        with pytest.raises(RegistryError):
            handler_registry.get(HANDLER_TYPE_HTTP)

        # Re-register with different handler
        handler_registry.register(HANDLER_TYPE_HTTP, MockDbHandler)
        assert handler_registry.get(HANDLER_TYPE_HTTP) is MockDbHandler

    def test_all_handler_types_registration(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test registering all defined handler types."""
        handlers = [
            (HANDLER_TYPE_HTTP, MockHttpHandler),
            (HANDLER_TYPE_DATABASE, MockDbHandler),
            (HANDLER_TYPE_KAFKA, MockKafkaHandler),
            (HANDLER_TYPE_VAULT, MockVaultHandler),
            (HANDLER_TYPE_CONSUL, MockConsulHandler),
            (HANDLER_TYPE_VALKEY, MockValkeyHandler),
            (HANDLER_TYPE_GRPC, MockGrpcHandler),
        ]
        for proto, cls in handlers:
            handler_registry.register(proto, cls)

        assert len(handler_registry) == len(handlers)
        for proto, cls in handlers:
            assert handler_registry.get(proto) is cls


class TestEventBusRegistryIntegration:
    """Integration tests for EventBusBindingRegistry with real-world scenarios."""

    def test_full_registration_workflow(
        self, event_bus_registry: EventBusBindingRegistry
    ) -> None:
        """Test complete workflow for event bus registration."""
        # Register
        event_bus_registry.register(EVENT_BUS_INMEMORY, MockInMemoryEventBus)
        assert event_bus_registry.get(EVENT_BUS_INMEMORY) is MockInMemoryEventBus

        # List
        kinds = event_bus_registry.list_bus_kinds()
        assert EVENT_BUS_INMEMORY in kinds

        # Verify duplicate raises
        with pytest.raises(RuntimeHostError):
            event_bus_registry.register(EVENT_BUS_INMEMORY, MockAlternativeEventBus)

        # Register different kind
        event_bus_registry.register(EVENT_BUS_KAFKA, MockKafkaEventBus)
        assert event_bus_registry.get(EVENT_BUS_KAFKA) is MockKafkaEventBus
        assert len(event_bus_registry.list_bus_kinds()) == 2


# =============================================================================
# Operation Prefix Routing Tests (OMN-807)
# =============================================================================


class TestOperationPrefixRouting:
    """Tests for operation prefix routing validation (OMN-807)."""

    def test_registered_prefixes_route_correctly(
        self, populated_handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that registered prefixes resolve to correct handlers."""
        # populated_handler_registry has http, db, kafka registered
        assert populated_handler_registry.get(HANDLER_TYPE_HTTP) is MockHttpHandler
        assert populated_handler_registry.get(HANDLER_TYPE_DATABASE) is MockDbHandler
        assert populated_handler_registry.get(HANDLER_TYPE_KAFKA) is MockKafkaHandler

    def test_unknown_prefix_raises_registry_error(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that unknown prefix raises RegistryError.

        CRITICAL: This is the negative test case required by OMN-807.
        Operations like "lolnope.query" MUST fail with RegistryError.
        """
        # Register only http
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)

        # Unknown prefixes must raise RegistryError
        unknown_prefixes = ["lolnope", "unknown", "nonexistent", "fake"]

        for prefix in unknown_prefixes:
            with pytest.raises(RegistryError) as exc_info:
                handler_registry.get(prefix)

            error_msg = str(exc_info.value)
            assert "No handler registered" in error_msg
            assert prefix in error_msg

    def test_operation_prefix_extraction_pattern(
        self, populated_handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test the operation prefix extraction pattern used by runtime.

        Operations like "http.get" -> prefix "http"
        Operations like "db.query" -> prefix "db"
        """
        operations = [
            ("http.get", HANDLER_TYPE_HTTP, MockHttpHandler),
            ("http.post", HANDLER_TYPE_HTTP, MockHttpHandler),
            ("db.query", HANDLER_TYPE_DATABASE, MockDbHandler),
            ("db.execute", HANDLER_TYPE_DATABASE, MockDbHandler),
            ("kafka.produce", HANDLER_TYPE_KAFKA, MockKafkaHandler),
            ("kafka.consume", HANDLER_TYPE_KAFKA, MockKafkaHandler),
        ]

        for operation, expected_prefix, expected_handler in operations:
            # Extract prefix (same pattern as RuntimeHostProcess._handle_envelope)
            prefix = operation.split(".")[0]
            assert prefix == expected_prefix

            # Verify handler resolves
            handler_cls = populated_handler_registry.get(prefix)
            assert handler_cls is expected_handler

    def test_canonical_prefixes_documented(self) -> None:
        """Test that all canonical prefixes are documented as constants.

        FROZEN prefixes per OMN-807:
        - db: Database operations
        - http: HTTP REST operations
        - kafka: Kafka message operations
        - consul: Consul service discovery
        - vault: Vault secret management
        - valkey: Valkey (Redis-compatible) cache
        - grpc: gRPC protocol operations
        """
        canonical_prefixes = {
            HANDLER_TYPE_DATABASE: "db",
            HANDLER_TYPE_HTTP: "http",
            HANDLER_TYPE_KAFKA: "kafka",
            HANDLER_TYPE_CONSUL: "consul",
            HANDLER_TYPE_VAULT: "vault",
            HANDLER_TYPE_VALKEY: "valkey",
            HANDLER_TYPE_GRPC: "grpc",
        }

        for constant, expected_value in canonical_prefixes.items():
            assert constant == expected_value, (
                f"{constant} should be '{expected_value}'"
            )

    def test_all_canonical_prefixes_are_unique(self) -> None:
        """Test that canonical prefixes are unique (no duplicates)."""
        prefixes = [
            HANDLER_TYPE_DATABASE,
            HANDLER_TYPE_HTTP,
            HANDLER_TYPE_KAFKA,
            HANDLER_TYPE_CONSUL,
            HANDLER_TYPE_VAULT,
            HANDLER_TYPE_VALKEY,
            HANDLER_TYPE_GRPC,
        ]
        assert len(prefixes) == len(set(prefixes)), "Duplicate prefix detected"

    def test_unknown_operation_prefix_fails_dispatch(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test that operations with unknown prefixes fail during dispatch.

        Simulates the runtime dispatch pattern where operation like
        'lolnope.query' would be split to extract 'lolnope' prefix.
        """
        # Setup: Register only valid prefixes
        handler_registry.register(HANDLER_TYPE_HTTP, MockHttpHandler)
        handler_registry.register(HANDLER_TYPE_DATABASE, MockDbHandler)

        # Invalid operations that must fail
        invalid_operations = [
            "lolnope.query",
            "fake.execute",
            "unknown.request",
            "invalid.operation",
        ]

        for operation in invalid_operations:
            # Extract prefix (same pattern as runtime dispatch)
            prefix = operation.split(".")[0]

            # Must raise RegistryError for unknown prefix
            with pytest.raises(RegistryError) as exc_info:
                handler_registry.get(prefix)

            error_msg = str(exc_info.value)
            assert prefix in error_msg
            assert "No handler registered" in error_msg


# =============================================================================
# UnknownHandlerTypeError Tests
# =============================================================================


class TestUnknownHandlerTypeError:
    """Tests for UnknownHandlerTypeError exception class."""

    def test_error_is_importable(self) -> None:
        """Test UnknownHandlerTypeError can be imported."""
        assert UnknownHandlerTypeError is not None

    def test_error_inherits_from_runtime_host_error(self) -> None:
        """Test UnknownHandlerTypeError inherits from RuntimeHostError."""
        assert issubclass(UnknownHandlerTypeError, RuntimeHostError)

    def test_error_with_context(self) -> None:
        """Test UnknownHandlerTypeError with context."""
        context = ModelInfraErrorContext(
            operation="lolnope.query",
        )
        error = UnknownHandlerTypeError(
            "No handler registered for prefix: lolnope",
            context=context,
            prefix="lolnope",
            registered_prefixes=["db", "http"],
        )
        assert "lolnope" in str(error)

    def test_error_without_context(self) -> None:
        """Test UnknownHandlerTypeError without context."""
        error = UnknownHandlerTypeError(
            "No handler registered for prefix: unknown",
            prefix="unknown",
        )
        assert "unknown" in str(error)

    def test_error_extra_context_preserved(self) -> None:
        """Test that extra context kwargs are preserved in error."""
        error = UnknownHandlerTypeError(
            "Handler not found",
            prefix="fake",
            registered_prefixes=["db", "http", "kafka"],
            operation="fake.query",
        )
        # Error should preserve the context in model
        assert error.model.context.get("prefix") == "fake"
        assert error.model.context.get("registered_prefixes") == ["db", "http", "kafka"]
        assert error.model.context.get("operation") == "fake.query"

    def test_error_is_exception(self) -> None:
        """Test UnknownHandlerTypeError is an Exception."""
        error = UnknownHandlerTypeError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, RuntimeHostError)

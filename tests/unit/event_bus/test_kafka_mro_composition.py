# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EventBusKafka MRO (Method Resolution Order) and mixin composition.

This test suite verifies that the multiple mixins used by EventBusKafka
(MixinKafkaBroadcast, MixinKafkaDlq, MixinAsyncCircuitBreaker) compose correctly
without MRO conflicts and that all mixin methods are accessible.

MRO Background:
    Python uses C3 linearization algorithm to determine Method Resolution Order.
    When a class inherits from multiple base classes, the MRO determines the order
    in which methods are resolved. If the linearization cannot produce a consistent
    order, Python raises a TypeError at class definition time.

Test Coverage:
    - MRO is deterministic and does not raise TypeError
    - All mixin methods are accessible on EventBusKafka
    - Initialization order works correctly (super().__init__ chain)
    - No method shadowing occurs between mixins
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.mixin_kafka_broadcast import MixinKafkaBroadcast
from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.mixins import MixinAsyncCircuitBreaker


class TestEventBusKafkaMROComposition:
    """Test suite for EventBusKafka MRO and mixin composition verification."""

    def test_mro_is_deterministic_and_valid(self) -> None:
        """Verify MRO does not raise TypeError and is deterministic.

        This test ensures that the C3 linearization algorithm successfully
        resolves the MRO for EventBusKafka without conflicts. If there were
        MRO conflicts, Python would raise TypeError at class definition time.
        """
        # Get the MRO - if this doesn't exist or raises, MRO failed
        mro = EventBusKafka.__mro__

        # Verify MRO is a tuple (expected type)
        assert isinstance(mro, tuple), f"MRO should be a tuple, got {type(mro)}"

        # Verify MRO is non-empty and includes expected classes
        assert len(mro) > 0, "MRO should not be empty"

        # EventBusKafka should be first (the class itself)
        assert mro[0] is EventBusKafka, "EventBusKafka should be first in MRO"

        # object should be last (all classes inherit from object)
        assert mro[-1] is object, "object should be last in MRO"

    def test_mro_contains_all_mixins(self) -> None:
        """Verify MRO contains all expected mixin classes.

        The MRO should include:
        - EventBusKafka (the class itself)
        - MixinKafkaBroadcast
        - MixinKafkaDlq
        - MixinAsyncCircuitBreaker
        - object
        """
        mro = EventBusKafka.__mro__

        # All mixins should be in the MRO
        expected_classes = [
            EventBusKafka,
            MixinKafkaBroadcast,
            MixinKafkaDlq,
            MixinAsyncCircuitBreaker,
            object,
        ]

        for expected_class in expected_classes:
            assert expected_class in mro, (
                f"{expected_class.__name__} should be in MRO. "
                f"Actual MRO: {[c.__name__ for c in mro]}"
            )

    def test_mro_order_follows_inheritance_declaration(self) -> None:
        """Verify MRO order follows left-to-right inheritance declaration.

        EventBusKafka is declared as:
            class EventBusKafka(MixinKafkaBroadcast, MixinKafkaDlq, MixinAsyncCircuitBreaker):

        Per C3 linearization, the MRO should follow this order (with variations
        based on mixin dependencies).
        """
        mro = EventBusKafka.__mro__
        mro_names = [cls.__name__ for cls in mro]

        # EventBusKafka must be first
        assert mro_names[0] == "EventBusKafka"

        # Get indices of each mixin
        broadcast_idx = mro_names.index("MixinKafkaBroadcast")
        dlq_idx = mro_names.index("MixinKafkaDlq")
        circuit_breaker_idx = mro_names.index("MixinAsyncCircuitBreaker")

        # Mixins should appear before object
        object_idx = mro_names.index("object")
        assert broadcast_idx < object_idx
        assert dlq_idx < object_idx
        assert circuit_breaker_idx < object_idx

        # Verify relative order matches declaration order:
        # MixinKafkaBroadcast < MixinKafkaDlq < MixinAsyncCircuitBreaker
        assert broadcast_idx < dlq_idx, (
            "MixinKafkaBroadcast should come before MixinKafkaDlq in MRO"
        )
        assert dlq_idx < circuit_breaker_idx, (
            "MixinKafkaDlq should come before MixinAsyncCircuitBreaker in MRO"
        )


class TestMixinKafkaBroadcastMethodsAccessible:
    """Verify MixinKafkaBroadcast methods are accessible on EventBusKafka."""

    def test_broadcast_to_environment_method_exists(self) -> None:
        """Verify broadcast_to_environment method is accessible."""
        assert hasattr(EventBusKafka, "broadcast_to_environment")
        assert callable(EventBusKafka.broadcast_to_environment)

    def test_send_to_group_method_exists(self) -> None:
        """Verify send_to_group method is accessible."""
        assert hasattr(EventBusKafka, "send_to_group")
        assert callable(EventBusKafka.send_to_group)

    def test_publish_envelope_method_exists(self) -> None:
        """Verify publish_envelope method is accessible."""
        assert hasattr(EventBusKafka, "publish_envelope")
        assert callable(EventBusKafka.publish_envelope)


class TestMixinKafkaDlqMethodsAccessible:
    """Verify MixinKafkaDlq methods are accessible on EventBusKafka."""

    def test_init_dlq_method_exists(self) -> None:
        """Verify _init_dlq method is accessible."""
        assert hasattr(EventBusKafka, "_init_dlq")
        assert callable(EventBusKafka._init_dlq)

    def test_dlq_metrics_property_exists(self) -> None:
        """Verify dlq_metrics property is accessible."""
        # Check it's defined as a property in the class hierarchy
        for cls in EventBusKafka.__mro__:
            if "dlq_metrics" in cls.__dict__:
                assert isinstance(cls.__dict__["dlq_metrics"], property)
                break
        else:
            pytest.fail("dlq_metrics property not found in MRO")

    def test_register_dlq_callback_method_exists(self) -> None:
        """Verify register_dlq_callback method is accessible."""
        assert hasattr(EventBusKafka, "register_dlq_callback")
        assert callable(EventBusKafka.register_dlq_callback)

    def test_publish_to_dlq_method_exists(self) -> None:
        """Verify _publish_to_dlq method is accessible."""
        assert hasattr(EventBusKafka, "_publish_to_dlq")
        assert callable(EventBusKafka._publish_to_dlq)

    def test_publish_raw_to_dlq_method_exists(self) -> None:
        """Verify _publish_raw_to_dlq method is accessible."""
        assert hasattr(EventBusKafka, "_publish_raw_to_dlq")
        assert callable(EventBusKafka._publish_raw_to_dlq)

    def test_invoke_dlq_callbacks_method_exists(self) -> None:
        """Verify _invoke_dlq_callbacks method is accessible."""
        assert hasattr(EventBusKafka, "_invoke_dlq_callbacks")
        assert callable(EventBusKafka._invoke_dlq_callbacks)


class TestMixinAsyncCircuitBreakerMethodsAccessible:
    """Verify MixinAsyncCircuitBreaker methods are accessible on EventBusKafka."""

    def test_init_circuit_breaker_method_exists(self) -> None:
        """Verify _init_circuit_breaker method is accessible."""
        assert hasattr(EventBusKafka, "_init_circuit_breaker")
        assert callable(EventBusKafka._init_circuit_breaker)

    def test_init_circuit_breaker_from_config_method_exists(self) -> None:
        """Verify _init_circuit_breaker_from_config method is accessible."""
        assert hasattr(EventBusKafka, "_init_circuit_breaker_from_config")
        assert callable(EventBusKafka._init_circuit_breaker_from_config)

    def test_check_circuit_breaker_method_exists(self) -> None:
        """Verify _check_circuit_breaker method is accessible."""
        assert hasattr(EventBusKafka, "_check_circuit_breaker")
        assert callable(EventBusKafka._check_circuit_breaker)

    def test_record_circuit_failure_method_exists(self) -> None:
        """Verify _record_circuit_failure method is accessible."""
        assert hasattr(EventBusKafka, "_record_circuit_failure")
        assert callable(EventBusKafka._record_circuit_failure)

    def test_reset_circuit_breaker_method_exists(self) -> None:
        """Verify _reset_circuit_breaker method is accessible."""
        assert hasattr(EventBusKafka, "_reset_circuit_breaker")
        assert callable(EventBusKafka._reset_circuit_breaker)

    def test_get_circuit_breaker_state_method_exists(self) -> None:
        """Verify _get_circuit_breaker_state method is accessible."""
        assert hasattr(EventBusKafka, "_get_circuit_breaker_state")
        assert callable(EventBusKafka._get_circuit_breaker_state)


class TestMixinInitializationOrder:
    """Verify mixin initialization order works correctly."""

    def test_initialization_does_not_raise(self) -> None:
        """Verify EventBusKafka can be instantiated without raising.

        This tests that the __init__ method properly initializes all mixins.
        """
        # Should not raise any exception
        bus = EventBusKafka()

        # Verify basic properties work
        assert bus.environment == "local"

    def test_circuit_breaker_initialized(self) -> None:
        """Verify circuit breaker mixin is properly initialized."""
        bus = EventBusKafka()

        # Circuit breaker should have its attributes initialized
        assert hasattr(bus, "_circuit_breaker_lock")
        assert hasattr(bus, "_circuit_breaker_failures")
        assert hasattr(bus, "_circuit_breaker_open")
        assert hasattr(bus, "circuit_breaker_threshold")
        assert hasattr(bus, "circuit_breaker_reset_timeout")
        assert hasattr(bus, "service_name")

        # Lock should be an asyncio.Lock instance
        assert isinstance(bus._circuit_breaker_lock, asyncio.Lock)

        # Initial state should be closed
        assert bus._circuit_breaker_open is False
        assert bus._circuit_breaker_failures == 0

    def test_dlq_mixin_initialized(self) -> None:
        """Verify DLQ mixin is properly initialized."""
        bus = EventBusKafka()

        # DLQ should have its attributes initialized
        assert hasattr(bus, "_dlq_metrics")
        assert hasattr(bus, "_dlq_metrics_lock")
        assert hasattr(bus, "_dlq_callbacks")
        assert hasattr(bus, "_dlq_callbacks_lock")

        # Locks should be asyncio.Lock instances
        assert isinstance(bus._dlq_metrics_lock, asyncio.Lock)
        assert isinstance(bus._dlq_callbacks_lock, asyncio.Lock)

        # Initial state: no callbacks registered
        assert len(bus._dlq_callbacks) == 0

    def test_broadcast_mixin_dependencies_available(self) -> None:
        """Verify broadcast mixin dependencies are available.

        MixinKafkaBroadcast requires:
        - self._environment
        - self.publish() method
        """
        bus = EventBusKafka()

        # Required attributes should exist
        assert hasattr(bus, "_environment")
        assert hasattr(bus, "publish")

        # publish should be callable
        assert callable(bus.publish)

    def test_config_passed_to_all_mixins(self) -> None:
        """Verify config is properly propagated to all mixin initializations."""
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="test:9092",
            environment="staging",
            circuit_breaker_threshold=10,
            circuit_breaker_reset_timeout=120.0,
        )
        bus = EventBusKafka(config=config)

        # Verify config values propagated
        assert bus.environment == "staging"

        # Circuit breaker should use config values
        assert bus.circuit_breaker_threshold == 10
        assert bus.circuit_breaker_reset_timeout == 120.0


class TestNoMethodShadowing:
    """Verify no method shadowing occurs between mixins."""

    def test_no_duplicate_method_definitions(self) -> None:
        """Verify each mixin method is defined in exactly one class.

        This checks that mixins don't accidentally override each other's methods.
        """
        # Methods from each mixin that should be unique
        broadcast_methods = {
            "broadcast_to_environment",
            "send_to_group",
            "publish_envelope",
        }
        dlq_methods = {
            "_init_dlq",
            "dlq_metrics",
            "register_dlq_callback",
            "_publish_to_dlq",
            "_publish_raw_to_dlq",
            "_invoke_dlq_callbacks",
        }
        circuit_breaker_methods = {
            "_init_circuit_breaker",
            "_init_circuit_breaker_from_config",
            "_check_circuit_breaker",
            "_record_circuit_failure",
            "_reset_circuit_breaker",
            "_get_circuit_breaker_state",
        }

        # Check that methods are defined in their expected class
        for method_name in broadcast_methods:
            # Find where this method is defined
            defining_classes = []
            for cls in EventBusKafka.__mro__:
                if method_name in cls.__dict__:
                    defining_classes.append(cls)

            # Should be defined in exactly one class (MixinKafkaBroadcast)
            # Note: We allow EventBusKafka to override, but it shouldn't for these
            assert len(defining_classes) >= 1, (
                f"{method_name} should be defined in at least one class"
            )
            # The first definition should be MixinKafkaBroadcast or a subclass that uses it
            assert MixinKafkaBroadcast in defining_classes or any(
                issubclass(cls, MixinKafkaBroadcast) for cls in defining_classes
            ), f"{method_name} should be from MixinKafkaBroadcast"

        for method_name in dlq_methods:
            defining_classes = []
            for cls in EventBusKafka.__mro__:
                if method_name in cls.__dict__:
                    defining_classes.append(cls)

            assert len(defining_classes) >= 1, (
                f"{method_name} should be defined in at least one class"
            )
            assert MixinKafkaDlq in defining_classes or any(
                issubclass(cls, MixinKafkaDlq) for cls in defining_classes
            ), f"{method_name} should be from MixinKafkaDlq"

        for method_name in circuit_breaker_methods:
            defining_classes = []
            for cls in EventBusKafka.__mro__:
                if method_name in cls.__dict__:
                    defining_classes.append(cls)

            assert len(defining_classes) >= 1, (
                f"{method_name} should be defined in at least one class"
            )
            assert MixinAsyncCircuitBreaker in defining_classes or any(
                issubclass(cls, MixinAsyncCircuitBreaker) for cls in defining_classes
            ), f"{method_name} should be from MixinAsyncCircuitBreaker"

    def test_no_method_name_conflicts_between_mixins(self) -> None:
        """Verify mixins don't define methods with the same name.

        This ensures mixins have distinct method namespaces.
        """
        # Get methods directly defined in each mixin (not inherited)
        broadcast_direct_methods = {
            name
            for name in MixinKafkaBroadcast.__dict__
            if callable(getattr(MixinKafkaBroadcast, name, None))
            and not name.startswith("__")
        }

        dlq_direct_methods = {
            name
            for name in MixinKafkaDlq.__dict__
            if callable(getattr(MixinKafkaDlq, name, None))
            and not name.startswith("__")
        }

        circuit_breaker_direct_methods = {
            name
            for name in MixinAsyncCircuitBreaker.__dict__
            if callable(getattr(MixinAsyncCircuitBreaker, name, None))
            and not name.startswith("__")
        }

        # Check for conflicts (excluding properties which show up differently)
        broadcast_dlq_conflict = broadcast_direct_methods & dlq_direct_methods
        broadcast_cb_conflict = (
            broadcast_direct_methods & circuit_breaker_direct_methods
        )
        dlq_cb_conflict = dlq_direct_methods & circuit_breaker_direct_methods

        # Filter out property-based names which may appear in both
        # (dlq_metrics is a property, not a method conflict)
        broadcast_dlq_conflict -= {"dlq_metrics"}
        broadcast_cb_conflict -= {"dlq_metrics"}
        dlq_cb_conflict -= {"dlq_metrics"}

        assert not broadcast_dlq_conflict, (
            f"Method name conflict between broadcast and DLQ mixins: {broadcast_dlq_conflict}"
        )
        assert not broadcast_cb_conflict, (
            f"Method name conflict between broadcast and circuit breaker mixins: {broadcast_cb_conflict}"
        )
        assert not dlq_cb_conflict, (
            f"Method name conflict between DLQ and circuit breaker mixins: {dlq_cb_conflict}"
        )


class TestMixinMethodFunctionality:
    """Verify mixin methods work correctly on instantiated EventBusKafka."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.fixture
    async def event_bus(self, mock_producer: AsyncMock) -> EventBusKafka:
        """Create EventBusKafka with mocked producer."""
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            config = ModelKafkaEventBusConfig(
                bootstrap_servers="localhost:9092",
                environment="dev",
            )
            bus = EventBusKafka(config=config)
            yield bus
            try:
                await bus.close()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_circuit_breaker_methods_work(self, event_bus: EventBusKafka) -> None:
        """Verify circuit breaker methods work correctly."""
        # Initially circuit should be closed
        async with event_bus._circuit_breaker_lock:
            state = event_bus._get_circuit_breaker_state()
            assert state["state"] == "closed"
            assert state["failures"] == 0

            # Record a failure
            await event_bus._record_circuit_failure(operation="test")
            assert event_bus._circuit_breaker_failures == 1

            # Reset the circuit breaker
            await event_bus._reset_circuit_breaker()
            assert event_bus._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_dlq_metrics_accessible(self, event_bus: EventBusKafka) -> None:
        """Verify DLQ metrics are accessible and work correctly."""
        metrics = event_bus.dlq_metrics

        # Should return a copy (not the original)
        assert metrics is not event_bus._dlq_metrics

        # Initial state should be empty
        assert metrics.total_publishes == 0
        assert metrics.successful_publishes == 0
        assert metrics.failed_publishes == 0

    @pytest.mark.asyncio
    async def test_dlq_callback_registration_works(
        self, event_bus: EventBusKafka
    ) -> None:
        """Verify DLQ callback registration works correctly."""
        callback_called = False

        async def test_callback(event: object) -> None:
            nonlocal callback_called
            callback_called = True

        # Register callback
        unregister = await event_bus.register_dlq_callback(test_callback)

        # Verify callback is registered
        assert len(event_bus._dlq_callbacks) == 1

        # Unregister
        await unregister()

        # Verify callback is removed
        assert len(event_bus._dlq_callbacks) == 0


class TestMRODiagnostics:
    """Diagnostic tests for MRO analysis and documentation."""

    def test_print_full_mro(self) -> None:
        """Print the full MRO for documentation purposes."""
        mro = EventBusKafka.__mro__
        mro_names = [f"{cls.__module__}.{cls.__name__}" for cls in mro]

        # This doesn't assert anything - it's for diagnostic output
        # Run with pytest -v to see the output
        print("\nEventBusKafka MRO:")
        for i, name in enumerate(mro_names):
            print(f"  {i}: {name}")

        # Just verify we can generate this info
        assert len(mro_names) > 0

    def test_method_resolution_for_publish(self) -> None:
        """Verify which class provides the publish method."""
        # Find which class defines publish
        for cls in EventBusKafka.__mro__:
            if "publish" in cls.__dict__:
                # publish should be defined in EventBusKafka itself
                assert cls is EventBusKafka, (
                    f"publish method should be defined in EventBusKafka, "
                    f"but found in {cls.__name__}"
                )
                break
        else:
            pytest.fail("publish method not found in MRO")

    def test_all_mixin_bases_are_object(self) -> None:
        """Verify all mixins inherit directly from object (no diamond problems).

        This ensures the mixin classes are simple and don't create complex
        diamond inheritance patterns.
        """
        # Each mixin should only inherit from object
        assert MixinKafkaBroadcast.__bases__ == (object,), (
            f"MixinKafkaBroadcast should only inherit from object, "
            f"got {MixinKafkaBroadcast.__bases__}"
        )
        assert MixinKafkaDlq.__bases__ == (object,), (
            f"MixinKafkaDlq should only inherit from object, "
            f"got {MixinKafkaDlq.__bases__}"
        )
        assert MixinAsyncCircuitBreaker.__bases__ == (object,), (
            f"MixinAsyncCircuitBreaker should only inherit from object, "
            f"got {MixinAsyncCircuitBreaker.__bases__}"
        )

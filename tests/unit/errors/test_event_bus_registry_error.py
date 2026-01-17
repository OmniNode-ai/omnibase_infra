# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EventBusRegistryError.

Tests follow TDD approach:
1. Write tests first (red phase)
2. Implement error class (green phase)
3. Refactor if needed (refactor phase)

All tests validate:
- Error class instantiation
- Inheritance chain
- Structured context fields via ModelInfraErrorContext
- Domain-specific context fields (bus_kind, bus_class, etc.)
- Integration with EventBusBindingRegistry
"""

from uuid import uuid4

import pytest

from omnibase_core.errors import ModelOnexError
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    EventBusRegistryError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.runtime.registry.registry_event_bus_binding import (
    EventBusBindingRegistry,
)


class TestEventBusRegistryErrorBasic:
    """Basic instantiation tests for EventBusRegistryError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation with message only."""
        error = EventBusRegistryError("Event bus not registered")
        assert "Event bus not registered" in str(error)
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)

    def test_with_bus_kind(self) -> None:
        """Test error with bus_kind context."""
        error = EventBusRegistryError(
            "Event bus kind not found",
            bus_kind="kafka",
        )
        assert error.model.context["bus_kind"] == "kafka"

    def test_with_bus_class(self) -> None:
        """Test error with bus_class context."""
        error = EventBusRegistryError(
            "Event bus class is invalid",
            bus_kind="custom",
            bus_class="InvalidEventBus",
        )
        assert error.model.context["bus_kind"] == "custom"
        assert error.model.context["bus_class"] == "InvalidEventBus"

    def test_with_available_kinds(self) -> None:
        """Test error with available_kinds context for not-found scenarios."""
        error = EventBusRegistryError(
            "Event bus kind 'unknown' is not registered",
            bus_kind="unknown",
            available_kinds=["inmemory", "kafka"],
        )
        assert error.model.context["bus_kind"] == "unknown"
        assert error.model.context["available_kinds"] == ["inmemory", "kafka"]

    def test_with_existing_class(self) -> None:
        """Test error with existing_class context for duplicate registration."""
        error = EventBusRegistryError(
            "Event bus kind 'inmemory' is already registered",
            bus_kind="inmemory",
            existing_class="InMemoryEventBus",
        )
        assert error.model.context["bus_kind"] == "inmemory"
        assert error.model.context["existing_class"] == "InMemoryEventBus"


class TestEventBusRegistryErrorWithContext:
    """Tests for EventBusRegistryError with ModelInfraErrorContext."""

    def test_with_context_model(self) -> None:
        """Test error with full context model."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="register_bus",
            correlation_id=correlation_id,
        )
        error = EventBusRegistryError(
            "Failed to register event bus",
            bus_kind="kafka",
            context=context,
        )
        assert error.model.correlation_id == correlation_id
        assert error.model.context["transport_type"] == EnumInfraTransportType.KAFKA
        assert error.model.context["operation"] == "register_bus"
        assert error.model.context["bus_kind"] == "kafka"

    def test_with_context_and_extra_fields(self) -> None:
        """Test error with context model and additional domain-specific fields."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="get_bus",
            correlation_id=correlation_id,
        )
        error = EventBusRegistryError(
            "Event bus not found",
            bus_kind="custom",
            available_kinds=["inmemory"],
            context=context,
        )
        assert error.model.context["transport_type"] == EnumInfraTransportType.RUNTIME
        assert error.model.context["bus_kind"] == "custom"
        assert error.model.context["available_kinds"] == ["inmemory"]


class TestEventBusRegistryErrorInheritance:
    """Tests for EventBusRegistryError inheritance chain."""

    def test_inherits_from_runtime_host_error(self) -> None:
        """Test that EventBusRegistryError inherits from RuntimeHostError."""
        error = EventBusRegistryError("test error")
        assert isinstance(error, RuntimeHostError)

    def test_inherits_from_model_onex_error(self) -> None:
        """Test that EventBusRegistryError inherits from ModelOnexError."""
        error = EventBusRegistryError("test error")
        assert isinstance(error, ModelOnexError)

    def test_inherits_from_exception(self) -> None:
        """Test that EventBusRegistryError inherits from Exception."""
        error = EventBusRegistryError("test error")
        assert isinstance(error, Exception)


class TestEventBusRegistryErrorChaining:
    """Tests for error chaining with EventBusRegistryError."""

    def test_error_chaining_preserves_cause(self) -> None:
        """Test that error chaining preserves the original exception."""
        original = ValueError("Original error")
        try:
            raise EventBusRegistryError("Wrapped error") from original
        except EventBusRegistryError as e:
            assert e.__cause__ == original
            assert isinstance(e.__cause__, ValueError)

    def test_error_chaining_with_context(self) -> None:
        """Test error chaining with context preserved."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            correlation_id=correlation_id,
        )
        original = TypeError("Bus class is not valid")
        try:
            raise EventBusRegistryError(
                "Failed to register bus",
                bus_kind="invalid",
                bus_class="NotABus",
                context=context,
            ) from original
        except EventBusRegistryError as e:
            assert e.__cause__ == original
            assert e.model.correlation_id == correlation_id
            assert e.model.context["bus_kind"] == "invalid"


class TestEventBusRegistryErrorIntegration:
    """Integration tests with EventBusBindingRegistry."""

    def test_registry_raises_error_on_get_unregistered(self) -> None:
        """Test that registry raises EventBusRegistryError on get unregistered."""
        registry = EventBusBindingRegistry()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.get("nonexistent")

        error = exc_info.value
        assert "nonexistent" in str(error)
        assert error.model.context["bus_kind"] == "nonexistent"
        assert error.model.context["available_kinds"] == []

    def test_registry_raises_error_on_duplicate_registration(self) -> None:
        """Test that registry raises EventBusRegistryError on duplicate registration."""

        class MockEventBus:
            async def publish_envelope(self, envelope: object, topic: str) -> None:
                pass

        registry = EventBusBindingRegistry()
        registry.register("test", MockEventBus)  # type: ignore[arg-type]

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", MockEventBus)  # type: ignore[arg-type]

        error = exc_info.value
        assert "already registered" in str(error)
        assert error.model.context["bus_kind"] == "test"
        assert error.model.context["existing_class"] == "MockEventBus"

    def test_registry_raises_error_on_invalid_protocol(self) -> None:
        """Test that registry raises EventBusRegistryError for invalid protocol."""

        class InvalidBus:
            """Bus without publish methods."""

        registry = EventBusBindingRegistry()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("invalid", InvalidBus)  # type: ignore[arg-type]

        error = exc_info.value
        assert "missing" in str(error)
        assert error.model.context["bus_kind"] == "invalid"
        assert error.model.context["bus_class"] == "InvalidBus"

    def test_registry_raises_error_on_non_callable_method(self) -> None:
        """Test that registry raises EventBusRegistryError for non-callable publish."""

        class InvalidBusNonCallable:
            publish_envelope = "not_callable"

        registry = EventBusBindingRegistry()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("invalid", InvalidBusNonCallable)  # type: ignore[arg-type]

        error = exc_info.value
        assert "not callable" in str(error)
        assert error.model.context["bus_kind"] == "invalid"
        assert error.model.context["bus_class"] == "InvalidBusNonCallable"


class TestEventBusRegistryErrorExtraContext:
    """Tests for extra context kwargs support."""

    def test_extra_context_kwargs(self) -> None:
        """Test that extra context kwargs are preserved."""
        error = EventBusRegistryError(
            "Custom error",
            bus_kind="custom",
            custom_field="custom_value",
            retry_count=3,
        )
        assert error.model.context["bus_kind"] == "custom"
        assert error.model.context["custom_field"] == "custom_value"
        assert error.model.context["retry_count"] == 3

    def test_none_fields_not_added(self) -> None:
        """Test that None-valued domain fields are not added to context."""
        error = EventBusRegistryError(
            "Test error",
            bus_kind=None,
            bus_class=None,
            available_kinds=None,
            existing_class=None,
        )
        # None values should not be in context
        assert "bus_kind" not in error.model.context
        assert "bus_class" not in error.model.context
        assert "available_kinds" not in error.model.context
        assert "existing_class" not in error.model.context


class TestEventBusRegistryErrorCorrelationId:
    """Tests for auto-generated correlation_id in registry errors (OMN-129).

    PR #129 requested adding correlation IDs to all registry errors for
    better observability and debugging. These tests verify that all error
    paths include a correlation_id.
    """

    def test_register_protocol_validation_error_has_correlation_id(self) -> None:
        """Test that protocol validation error includes correlation_id."""

        class InvalidBus:
            """Bus without publish methods."""

        registry = EventBusBindingRegistry()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("invalid", InvalidBus)  # type: ignore[arg-type]

        error = exc_info.value
        # Verify correlation_id is auto-generated
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_register_non_callable_error_has_correlation_id(self) -> None:
        """Test that non-callable method error includes correlation_id."""

        class InvalidBusNonCallable:
            publish_envelope = "not_callable"

        registry = EventBusBindingRegistry()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("invalid", InvalidBusNonCallable)  # type: ignore[arg-type]

        error = exc_info.value
        # Verify correlation_id is auto-generated
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_register_duplicate_error_has_correlation_id(self) -> None:
        """Test that duplicate registration error includes correlation_id."""

        class MockEventBus:
            async def publish_envelope(self, envelope: object, topic: str) -> None:
                pass

        registry = EventBusBindingRegistry()
        registry.register("test", MockEventBus)  # type: ignore[arg-type]

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", MockEventBus)  # type: ignore[arg-type]

        error = exc_info.value
        # Verify correlation_id is auto-generated
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_get_unregistered_error_has_correlation_id(self) -> None:
        """Test that get unregistered error includes correlation_id."""
        registry = EventBusBindingRegistry()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.get("nonexistent")

        error = exc_info.value
        # Verify correlation_id is auto-generated
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "get"

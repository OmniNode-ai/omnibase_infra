# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for correlation_id in registry errors (PR #129).

PR #129 requested adding correlation IDs to all registry errors for better
observability and debugging. These tests verify that all registry error paths
include auto-generated correlation_ids for traceability.

Registry classes covered:
- RegistryProtocolBinding (registry_protocol_binding.py)
- RegistryEventBusBinding (registry_event_bus_binding.py)
- RegistryCompute (registry_compute.py)
- RegistryMessageType (registry_message_type.py) - via MessageTypeRegistryError
"""

from datetime import UTC

import pytest

from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.errors import (
    ComputeRegistryError,
    EventBusRegistryError,
)
from omnibase_infra.models.registry.model_domain_constraint import (
    ModelDomainConstraint,
)
from omnibase_infra.models.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)
from omnibase_infra.runtime.models import ModelComputeRegistration
from omnibase_infra.runtime.registry.registry_event_bus_binding import (
    RegistryEventBusBinding,
)
from omnibase_infra.runtime.registry.registry_message_type import (
    MessageTypeRegistryError,
    RegistryMessageType,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    RegistryError,
    RegistryProtocolBinding,
)
from omnibase_infra.runtime.registry_compute import RegistryCompute


class TestProtocolBindingRegistryCorrelationId:
    """Tests for correlation_id in RegistryProtocolBinding errors."""

    def test_register_missing_execute_method_has_correlation_id(self) -> None:
        """Test that missing execute method error includes correlation_id."""

        class InvalidHandler:
            """Handler without execute method."""

        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandler)  # type: ignore[arg-type]

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_register_non_callable_execute_has_correlation_id(self) -> None:
        """Test that non-callable execute error includes correlation_id."""

        class InvalidHandlerNonCallable:
            execute = "not_callable"

        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandlerNonCallable)  # type: ignore[arg-type]

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_get_unregistered_has_correlation_id(self) -> None:
        """Test that get unregistered error includes correlation_id."""
        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.get("nonexistent")

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "get"


class TestEventBusBindingRegistryCorrelationId:
    """Tests for correlation_id in RegistryEventBusBinding errors."""

    def test_register_missing_publish_has_correlation_id(self) -> None:
        """Test that missing publish method error includes correlation_id."""

        class InvalidBus:
            """Bus without publish methods."""

        registry = RegistryEventBusBinding()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", InvalidBus)  # type: ignore[arg-type]

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_register_non_callable_publish_has_correlation_id(self) -> None:
        """Test that non-callable publish error includes correlation_id."""

        class InvalidBusNonCallable:
            publish_envelope = "not_callable"

        registry = RegistryEventBusBinding()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", InvalidBusNonCallable)  # type: ignore[arg-type]

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_register_duplicate_has_correlation_id(self) -> None:
        """Test that duplicate registration error includes correlation_id."""

        class MockEventBus:
            async def publish_envelope(
                self, envelope: object, topic: str, *, key: bytes | None = None
            ) -> None:
                pass

        registry = RegistryEventBusBinding()
        registry.register("test", MockEventBus)  # type: ignore[arg-type]

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", MockEventBus)  # type: ignore[arg-type]

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_get_unregistered_has_correlation_id(self) -> None:
        """Test that get unregistered error includes correlation_id."""
        registry = RegistryEventBusBinding()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.get("nonexistent")

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "get"


class TestRegistryComputeCorrelationId:
    """Tests for correlation_id in RegistryCompute errors."""

    def test_register_missing_execute_has_correlation_id(self) -> None:
        """Test that missing execute method error includes correlation_id."""

        class InvalidPlugin:
            """Plugin without execute method."""

        registry = RegistryCompute()
        registration = ModelComputeRegistration(
            plugin_id="invalid",
            plugin_class=InvalidPlugin,  # type: ignore[arg-type]
        )

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.register(registration)

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_register_non_callable_execute_has_correlation_id(self) -> None:
        """Test that non-callable execute error includes correlation_id."""

        class InvalidPluginNonCallable:
            execute = "not_callable"

        registry = RegistryCompute()
        registration = ModelComputeRegistration(
            plugin_id="invalid",
            plugin_class=InvalidPluginNonCallable,  # type: ignore[arg-type]
        )

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.register(registration)

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "register"

    def test_register_async_without_flag_has_correlation_id(self) -> None:
        """Test that async without flag error includes correlation_id."""

        class AsyncPlugin:
            async def execute(self, input_data: object, context: object) -> object:
                return {}

        registry = RegistryCompute()
        registration = ModelComputeRegistration(
            plugin_id="async_plugin",
            plugin_class=AsyncPlugin,  # type: ignore[arg-type]
            deterministic_async=False,
        )

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.register(registration)

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "validate_sync_enforcement"

    def test_get_unregistered_has_correlation_id(self) -> None:
        """Test that get unregistered error includes correlation_id."""
        registry = RegistryCompute()

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.get("nonexistent")

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "get"

    def test_get_wrong_version_has_correlation_id(self) -> None:
        """Test that wrong version error includes correlation_id."""

        class ValidPlugin:
            def execute(self, input_data: object, context: object) -> object:
                return {}

        registry = RegistryCompute()
        registry.register_plugin("test", ValidPlugin)  # type: ignore[arg-type]

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.get("test", version="9.9.9")

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "get"

    def test_invalid_version_format_has_correlation_id(self) -> None:
        """Test that invalid version format error includes correlation_id.

        The version validation happens during semver parsing when retrieving
        a plugin (using _parse_semver for sorting).
        """

        class ValidPlugin:
            def execute(self, input_data: object, context: object) -> object:
                return {}

        registry = RegistryCompute()
        # Register with valid version first
        registry.register_plugin("test", ValidPlugin, version="1.0.0")  # type: ignore[arg-type]

        # Try to access with non-existent version
        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.get("test", version="invalid")

        error = exc_info.value
        assert error.model.correlation_id is not None
        # For version not found error
        assert error.model.context["operation"] == "get"


class TestMessageTypeRegistryCorrelationId:
    """Tests for correlation_id in RegistryMessageType errors.

    Note: RegistryMessageType uses MessageTypeRegistryError which now accepts
    context parameter with correlation_id. The error raises use auto-generated
    correlation_ids via ModelInfraErrorContext.with_correlation().
    """

    def test_get_handlers_not_found_accepts_context(self) -> None:
        """Test that MessageTypeRegistryError accepts context with correlation_id."""
        # First verify the error class accepts context
        from uuid import uuid4

        from omnibase_infra.errors import ModelInfraErrorContext
        from omnibase_infra.models.errors import ModelMessageTypeRegistryErrorContext

        context = ModelInfraErrorContext.with_correlation(
            correlation_id=uuid4(),
            operation="get_handlers",
        )
        registry_context = ModelMessageTypeRegistryErrorContext(
            message_type="Unknown",
        )
        error = MessageTypeRegistryError(
            "Message type not found",
            registry_context=registry_context,
            context=context,
        )
        assert error.model.correlation_id is not None
        assert error.model.context["operation"] == "get_handlers"

    def test_category_constraint_mismatch_error_structure(self) -> None:
        """Test MessageTypeRegistryError structure for constraint mismatch."""
        from datetime import datetime

        registry = RegistryMessageType()
        now = datetime.now(tz=UTC)

        # Register first entry
        entry1 = ModelMessageTypeEntry(
            message_type="TestEvent",
            handler_ids=("handler1",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="test"),
            registered_at=now,
        )
        registry.register_message_type(entry1)

        # Attempt to register conflicting entry
        entry2 = ModelMessageTypeEntry(
            message_type="TestEvent",
            handler_ids=("handler2",),
            allowed_categories=frozenset([EnumMessageCategory.COMMAND]),  # Different!
            domain_constraint=ModelDomainConstraint(owning_domain="test"),
            registered_at=now,
        )

        with pytest.raises(MessageTypeRegistryError) as exc_info:
            registry.register_message_type(entry2)

        error = exc_info.value
        # MessageTypeRegistryError should include message_type in context
        assert error.model.context["message_type"] == "TestEvent"

    def test_none_entry_registration_fails(self) -> None:
        """Test that registering None entry raises proper error."""
        registry = RegistryMessageType()

        with pytest.raises(ModelOnexError) as exc_info:
            registry.register_message_type(None)  # type: ignore[arg-type]

        error = exc_info.value
        assert "None" in str(error)

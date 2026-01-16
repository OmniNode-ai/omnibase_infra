# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for runtime type validation in registry classes.

This module tests that all registry classes properly validate registered
classes implement their required protocols at registration time.
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import (
    ComputeRegistryError,
    EventBusRegistryError,
    PolicyRegistryError,
)
from omnibase_infra.runtime.models import (
    ModelComputeRegistration,
    ModelPolicyRegistration,
)
from omnibase_infra.runtime.registry.registry_event_bus_binding import (
    RegistryEventBusBinding,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    RegistryError,
    RegistryProtocolBinding,
)
from omnibase_infra.runtime.registry_compute import RegistryCompute
from omnibase_infra.runtime.registry_policy import RegistryPolicy

# =============================================================================
# Test Classes - Invalid Implementations
# =============================================================================


class InvalidHandler:
    """Invalid handler class - missing both execute() and handle() methods."""


class InvalidHandlerExecuteNonCallable:
    """Invalid handler class - execute is not callable."""

    execute = "not_callable"  # type: ignore[assignment]


class InvalidHandlerHandleNonCallable:
    """Invalid handler class - handle is not callable."""

    handle = "not_callable"  # type: ignore[assignment]


class InvalidHandlerBothNonCallable:
    """Invalid handler class - both execute and handle are not callable."""

    execute = "not_callable"  # type: ignore[assignment]
    handle = "not_callable"  # type: ignore[assignment]


class InvalidEventBus:
    """Invalid event bus class - missing publish methods."""


class InvalidEventBusNonCallable:
    """Invalid event bus class - publish_envelope is not callable."""

    publish_envelope = "not_callable"  # type: ignore[assignment]


class InvalidEventBusPublishNonCallable:
    """Invalid event bus class - publish is not callable."""

    publish = "not_callable"  # type: ignore[assignment]


class InvalidPolicy:
    """Invalid policy class - missing required protocol methods."""


class InvalidPolicyMissingEvaluate:
    """Invalid policy class - missing evaluate() method."""

    @property
    def policy_id(self) -> str:
        return "test_policy"

    @property
    def policy_type(self) -> str:
        return "orchestrator"


class InvalidPolicyNonCallableEvaluate:
    """Invalid policy class - evaluate is not callable."""

    @property
    def policy_id(self) -> str:
        return "test_policy"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    evaluate = "not_callable"  # type: ignore[assignment]


class InvalidComputePlugin:
    """Invalid compute plugin class - missing execute() method."""


class InvalidComputePluginNonCallable:
    """Invalid compute plugin class - execute is not callable."""

    execute = "not_callable"  # type: ignore[assignment]


# =============================================================================
# Test Classes - Valid Implementations
# =============================================================================


class ValidHandler:
    """Valid handler class with execute() method."""

    def execute(self, request: object) -> object:
        return {"status": "ok"}


class ValidHandlerWithHandle:
    """Valid handler class with handle() method."""

    async def handle(self, envelope: object) -> object:
        return {"status": "ok"}


class ValidHandlerWithBoth:
    """Valid handler class with both execute() and handle() methods."""

    def execute(self, request: object) -> object:
        return {"status": "ok"}

    async def handle(self, envelope: object) -> object:
        return {"status": "ok"}


class ValidEventBus:
    """Valid event bus class with publish_envelope() method."""

    async def publish_envelope(self, envelope: object, topic: str) -> None:
        pass


class ValidEventBusWithPublish:
    """Valid event bus class with publish() method."""

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        pass


class ValidPolicy:
    """Valid policy class implementing ProtocolPolicy."""

    @property
    def policy_id(self) -> str:
        return "test_policy"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    def evaluate(self, context: object) -> object:
        return {"result": True}

    def decide(self, context: object) -> object:
        return self.evaluate(context)


class ValidComputePlugin:
    """Valid compute plugin class implementing ProtocolPluginCompute."""

    def execute(self, input_data: object, context: object) -> object:
        return {"result": "processed"}


# =============================================================================
# RegistryProtocolBinding Tests
# =============================================================================


class TestProtocolBindingRegistryValidation:
    """Test runtime type validation in RegistryProtocolBinding.

    This test class validates that RegistryProtocolBinding.register() performs
    proper runtime type checking following the RegistryEventBusBinding pattern:
    - Handler must have either execute() or handle() method (or both)
    - At least one handler method must be callable
    """

    def test_register_invalid_handler_missing_both_methods(self) -> None:
        """Test that registering handler without execute() or handle() raises error."""
        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandler)

        assert "missing 'execute()' or 'handle()' method" in str(exc_info.value)
        assert "InvalidHandler" in str(exc_info.value)

    def test_register_invalid_handler_non_callable_execute(self) -> None:
        """Test that registering handler with non-callable execute raises error."""
        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandlerExecuteNonCallable)

        assert "not callable" in str(exc_info.value)
        assert "'execute'" in str(exc_info.value)
        assert "InvalidHandlerExecuteNonCallable" in str(exc_info.value)

    def test_register_invalid_handler_non_callable_handle(self) -> None:
        """Test that registering handler with non-callable handle raises error."""
        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandlerHandleNonCallable)

        assert "not callable" in str(exc_info.value)
        assert "'handle'" in str(exc_info.value)
        assert "InvalidHandlerHandleNonCallable" in str(exc_info.value)

    def test_register_invalid_handler_both_non_callable(self) -> None:
        """Test that registering handler with both non-callable methods raises error."""
        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandlerBothNonCallable)

        # Should fail on first non-callable check (execute)
        assert "not callable" in str(exc_info.value)
        assert "InvalidHandlerBothNonCallable" in str(exc_info.value)

    def test_register_valid_handler_with_execute_succeeds(self) -> None:
        """Test that registering handler with execute() method succeeds."""
        registry = RegistryProtocolBinding()

        # Should not raise
        registry.register("test", ValidHandler)

        # Verify registration
        assert registry.is_registered("test")
        handler_cls = registry.get("test")
        assert handler_cls is ValidHandler

    def test_register_valid_handler_with_handle_succeeds(self) -> None:
        """Test that registering handler with only handle() method succeeds."""
        registry = RegistryProtocolBinding()

        # Should not raise - handle() alone is sufficient
        registry.register("test", ValidHandlerWithHandle)

        # Verify registration
        assert registry.is_registered("test")
        handler_cls = registry.get("test")
        assert handler_cls is ValidHandlerWithHandle

    def test_register_valid_handler_with_both_methods_succeeds(self) -> None:
        """Test that registering handler with both execute() and handle() succeeds."""
        registry = RegistryProtocolBinding()

        # Should not raise
        registry.register("test", ValidHandlerWithBoth)

        # Verify registration
        assert registry.is_registered("test")
        handler_cls = registry.get("test")
        assert handler_cls is ValidHandlerWithBoth

    def test_error_message_includes_protocol_type(self) -> None:
        """Test that error message includes the protocol type being registered."""
        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("custom-protocol", InvalidHandler)

        # Verify error context includes protocol type
        error = exc_info.value
        assert hasattr(error, "protocol_type") or "custom-protocol" in str(error)

    def test_error_message_includes_handler_class_name(self) -> None:
        """Test that error message includes the handler class name."""
        registry = RegistryProtocolBinding()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandler)

        assert "InvalidHandler" in str(exc_info.value)


# =============================================================================
# RegistryEventBusBinding Tests
# =============================================================================


class TestEventBusBindingRegistryValidation:
    """Test runtime type validation in RegistryEventBusBinding."""

    def test_register_invalid_event_bus_missing_methods(self) -> None:
        """Test that registering event bus without publish methods raises error."""
        registry = RegistryEventBusBinding()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", InvalidEventBus)

        assert "missing 'publish_envelope()' or 'publish()' method" in str(
            exc_info.value
        )
        assert "InvalidEventBus" in str(exc_info.value)

    def test_register_invalid_event_bus_non_callable(self) -> None:
        """Test that registering event bus with non-callable publish_envelope raises error."""
        registry = RegistryEventBusBinding()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", InvalidEventBusNonCallable)

        assert "not callable" in str(exc_info.value)
        assert "InvalidEventBusNonCallable" in str(exc_info.value)

    def test_register_invalid_event_bus_publish_non_callable(self) -> None:
        """Test that registering event bus with non-callable publish raises error."""
        registry = RegistryEventBusBinding()

        with pytest.raises(EventBusRegistryError) as exc_info:
            registry.register("test", InvalidEventBusPublishNonCallable)

        assert "not callable" in str(exc_info.value)
        assert "InvalidEventBusPublishNonCallable" in str(exc_info.value)

    def test_register_valid_event_bus_with_publish_envelope(self) -> None:
        """Test that registering valid event bus with publish_envelope succeeds."""
        registry = RegistryEventBusBinding()

        # Should not raise
        registry.register("test", ValidEventBus)

        # Verify registration
        assert registry.is_registered("test")
        bus_cls = registry.get("test")
        assert bus_cls is ValidEventBus

    def test_register_valid_event_bus_with_publish(self) -> None:
        """Test that registering valid event bus with publish succeeds."""
        registry = RegistryEventBusBinding()

        # Should not raise
        registry.register("test", ValidEventBusWithPublish)

        # Verify registration
        assert registry.is_registered("test")
        bus_cls = registry.get("test")
        assert bus_cls is ValidEventBusWithPublish


# =============================================================================
# RegistryPolicy Tests
# =============================================================================


class TestPolicyRegistryValidation:
    """Test runtime type validation in RegistryPolicy."""

    def test_register_invalid_policy_missing_all_methods(self) -> None:
        """Test that registering policy without required methods raises error."""
        registry = RegistryPolicy()

        registration = ModelPolicyRegistration(
            policy_id="test_policy",
            policy_class=InvalidPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        with pytest.raises(PolicyRegistryError) as exc_info:
            registry.register(registration)

        error_msg = str(exc_info.value)
        assert "does not implement ProtocolPolicy" in error_msg
        assert "policy_id property" in error_msg
        assert "policy_type property" in error_msg
        assert "evaluate() method" in error_msg

    def test_register_invalid_policy_missing_evaluate(self) -> None:
        """Test that registering policy without evaluate() method raises error."""
        registry = RegistryPolicy()

        registration = ModelPolicyRegistration(
            policy_id="test_policy",
            policy_class=InvalidPolicyMissingEvaluate,
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        with pytest.raises(PolicyRegistryError) as exc_info:
            registry.register(registration)

        assert "missing evaluate() method" in str(exc_info.value)
        assert "InvalidPolicyMissingEvaluate" in str(exc_info.value)

    def test_register_invalid_policy_non_callable_evaluate(self) -> None:
        """Test that registering policy with non-callable evaluate raises error."""
        registry = RegistryPolicy()

        registration = ModelPolicyRegistration(
            policy_id="test_policy",
            policy_class=InvalidPolicyNonCallableEvaluate,
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        with pytest.raises(PolicyRegistryError) as exc_info:
            registry.register(registration)

        assert "evaluate() method (not callable)" in str(exc_info.value)

    def test_register_valid_policy_succeeds(self) -> None:
        """Test that registering valid policy succeeds."""
        registry = RegistryPolicy()

        registration = ModelPolicyRegistration(
            policy_id="test_policy",
            policy_class=ValidPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        # Should not raise
        registry.register(registration)

        # Verify registration
        assert registry.is_registered("test_policy")
        policy_cls = registry.get("test_policy")
        assert policy_cls is ValidPolicy


# =============================================================================
# RegistryCompute Tests
# =============================================================================


class TestRegistryComputeValidation:
    """Test runtime type validation in RegistryCompute."""

    def test_register_invalid_compute_plugin_missing_execute(self) -> None:
        """Test that registering compute plugin without execute() raises error."""
        registry = RegistryCompute()

        registration = ModelComputeRegistration(
            plugin_id="test_plugin",
            plugin_class=InvalidComputePlugin,
        )

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.register(registration)

        assert "missing 'execute()' method" in str(exc_info.value)
        assert "InvalidComputePlugin" in str(exc_info.value)

    def test_register_invalid_compute_plugin_non_callable_execute(self) -> None:
        """Test that registering plugin with non-callable execute raises error."""
        registry = RegistryCompute()

        registration = ModelComputeRegistration(
            plugin_id="test_plugin",
            plugin_class=InvalidComputePluginNonCallable,
        )

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.register(registration)

        assert "not callable" in str(exc_info.value)
        assert "InvalidComputePluginNonCallable" in str(exc_info.value)

    def test_register_valid_compute_plugin_succeeds(self) -> None:
        """Test that registering valid compute plugin succeeds."""
        registry = RegistryCompute()

        registration = ModelComputeRegistration(
            plugin_id="test_plugin",
            plugin_class=ValidComputePlugin,
        )

        # Should not raise
        registry.register(registration)

        # Verify registration
        assert registry.is_registered("test_plugin")
        plugin_cls = registry.get("test_plugin")
        assert plugin_cls is ValidComputePlugin

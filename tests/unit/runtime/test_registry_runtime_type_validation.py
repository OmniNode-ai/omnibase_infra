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
    PolicyRegistryError,
    RuntimeHostError,
)
from omnibase_infra.runtime.models import (
    ModelComputeRegistration,
    ModelPolicyRegistration,
)
from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.registry.registry_event_bus_binding import (
    EventBusBindingRegistry,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    ProtocolBindingRegistry,
    RegistryError,
)
from omnibase_infra.runtime.registry_compute import RegistryCompute

# =============================================================================
# Test Classes - Invalid Implementations
# =============================================================================


class InvalidHandler:
    """Invalid handler class - missing execute() method."""


class InvalidHandlerNonCallable:
    """Invalid handler class - execute is not callable."""

    execute = "not_callable"  # type: ignore[assignment]


class InvalidEventBus:
    """Invalid event bus class - missing publish methods."""


class InvalidEventBusNonCallable:
    """Invalid event bus class - publish_envelope is not callable."""

    publish_envelope = "not_callable"  # type: ignore[assignment]


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
# ProtocolBindingRegistry Tests
# =============================================================================


class TestProtocolBindingRegistryValidation:
    """Test runtime type validation in ProtocolBindingRegistry."""

    def test_register_invalid_handler_missing_handle(self) -> None:
        """Test that registering handler without execute() method raises error."""
        registry = ProtocolBindingRegistry()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandler)

        assert "missing 'execute()' method" in str(exc_info.value)
        assert "InvalidHandler" in str(exc_info.value)

    def test_register_invalid_handler_non_callable_handle(self) -> None:
        """Test that registering handler with non-callable execute raises error."""
        registry = ProtocolBindingRegistry()

        with pytest.raises(RegistryError) as exc_info:
            registry.register("test", InvalidHandlerNonCallable)

        assert "not callable" in str(exc_info.value)
        assert "InvalidHandlerNonCallable" in str(exc_info.value)

    def test_register_valid_handler_succeeds(self) -> None:
        """Test that registering valid handler succeeds."""
        registry = ProtocolBindingRegistry()

        # Should not raise
        registry.register("test", ValidHandler)

        # Verify registration
        assert registry.is_registered("test")
        handler_cls = registry.get("test")
        assert handler_cls is ValidHandler


# =============================================================================
# EventBusBindingRegistry Tests
# =============================================================================


class TestEventBusBindingRegistryValidation:
    """Test runtime type validation in EventBusBindingRegistry."""

    def test_register_invalid_event_bus_missing_methods(self) -> None:
        """Test that registering event bus without publish methods raises error."""
        registry = EventBusBindingRegistry()

        with pytest.raises(RuntimeHostError) as exc_info:
            registry.register("test", InvalidEventBus)

        assert "missing 'publish_envelope()' or 'publish()' method" in str(
            exc_info.value
        )
        assert "InvalidEventBus" in str(exc_info.value)

    def test_register_invalid_event_bus_non_callable(self) -> None:
        """Test that registering event bus with non-callable publish raises error."""
        registry = EventBusBindingRegistry()

        with pytest.raises(RuntimeHostError) as exc_info:
            registry.register("test", InvalidEventBusNonCallable)

        assert "not callable" in str(exc_info.value)
        assert "InvalidEventBusNonCallable" in str(exc_info.value)

    def test_register_valid_event_bus_with_publish_envelope(self) -> None:
        """Test that registering valid event bus with publish_envelope succeeds."""
        registry = EventBusBindingRegistry()

        # Should not raise
        registry.register("test", ValidEventBus)

        # Verify registration
        assert registry.is_registered("test")
        bus_cls = registry.get("test")
        assert bus_cls is ValidEventBus

    def test_register_valid_event_bus_with_publish(self) -> None:
        """Test that registering valid event bus with publish succeeds."""
        registry = EventBusBindingRegistry()

        # Should not raise
        registry.register("test", ValidEventBusWithPublish)

        # Verify registration
        assert registry.is_registered("test")
        bus_cls = registry.get("test")
        assert bus_cls is ValidEventBusWithPublish


# =============================================================================
# PolicyRegistry Tests
# =============================================================================


class TestPolicyRegistryValidation:
    """Test runtime type validation in PolicyRegistry."""

    def test_register_invalid_policy_missing_all_methods(self) -> None:
        """Test that registering policy without required methods raises error."""
        registry = PolicyRegistry()

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
        registry = PolicyRegistry()

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
        registry = PolicyRegistry()

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
        registry = PolicyRegistry()

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

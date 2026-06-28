# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for PluginComputeBase abstract base class and protocol conformance.

Tests verify:
- Protocol conformance for base class
- Abstract base class behavior
- Validation hook execution order
- Context propagation

Note on Type Annotations:
    This test module intentionally uses dict types for test plugin implementations
    and passes raw dict inputs to plugin.execute() for testing purposes.
    The mypy directives below disable type errors for these intentional patterns.
"""
# mypy: disable-error-code="override, arg-type, attr-defined, index, return-value"

import pytest

from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase


class TestProtocolConformance:
    """Test that implementations conform to ProtocolPluginCompute.

    Per ONEX conventions, protocol conformance is verified via duck typing
    by checking for required method presence and callability, rather than
    using isinstance checks with Protocol types.
    """

    def test_protocol_conformance_with_base_class(self) -> None:
        """PluginComputeBase conforms to ProtocolPluginCompute."""

        # Arrange: Create concrete implementation of base class
        class ConcretePlugin(PluginComputeBase):
            def execute(self, input_data: dict, context: dict) -> dict:
                return input_data

        instance = ConcretePlugin()

        # Act & Assert: Verify protocol conformance via duck typing
        # ProtocolPluginCompute requires 'execute' method
        assert hasattr(instance, "execute"), "Must have 'execute' method"
        assert callable(instance.execute), "'execute' must be callable"


class TestBaseClassAbstraction:
    """Test abstract base class behavior."""

    def test_base_class_is_abstract(self) -> None:
        """Cannot instantiate PluginComputeBase directly."""
        # Arrange
        # Act & Assert: Attempting to instantiate should raise TypeError
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PluginComputeBase()  # type: ignore[abstract]

    def test_execute_method_is_abstract(self) -> None:
        """Must override execute() method."""

        # Arrange: Create class without execute() implementation
        class IncompletePlugin(PluginComputeBase):
            pass

        # Act & Assert: Attempting to instantiate should raise TypeError
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletePlugin()  # type: ignore[abstract]

    def test_validation_hooks_are_optional(self) -> None:
        """Can use base class without overriding validation hooks."""

        # Arrange: Create minimal implementation with only execute()
        class MinimalPlugin(PluginComputeBase):
            def execute(self, input_data: dict, context: dict) -> dict:
                return {"result": "success"}

        # Act: Instantiate and execute
        plugin = MinimalPlugin()
        result = plugin.execute({"input": "test"}, {"correlation_id": "test-123"})

        # Assert: Should work without validation hooks
        assert result == {"result": "success"}


class TestValidationHooks:
    """Test validation hook execution and behavior."""

    def test_validate_input_called_before_execute(self) -> None:
        """validate_input() hook must be called manually by external executor."""
        # Arrange: Track execution order
        execution_order: list[str] = []

        class TrackingPlugin(PluginComputeBase):
            def validate_input(self, input_data: dict) -> None:
                execution_order.append("validate_input")

            def execute(self, input_data: dict, context: dict) -> dict:
                execution_order.append("execute")
                return input_data

        plugin = TrackingPlugin()
        input_data = {"test": "data"}
        context = {"correlation_id": "test-123"}

        # Act: Manually call validation hook (simulating external executor)
        plugin.validate_input(input_data)
        plugin.execute(input_data, context)

        # Assert: validate_input called before execute
        assert execution_order == ["validate_input", "execute"]

    def test_validate_output_called_after_execute(self) -> None:
        """validate_output() hook must be called manually by external executor."""
        # Arrange: Track execution order
        execution_order: list[str] = []

        class TrackingPlugin(PluginComputeBase):
            def execute(self, input_data: dict, context: dict) -> dict:
                execution_order.append("execute")
                return {"result": "data"}

            def validate_output(self, output_data: dict) -> None:
                execution_order.append("validate_output")

        plugin = TrackingPlugin()
        input_data = {"test": "data"}
        context = {"correlation_id": "test-123"}

        # Act: Manually call validation hook (simulating external executor)
        output = plugin.execute(input_data, context)
        plugin.validate_output(output)

        # Assert: validate_output called after execute
        assert execution_order == ["execute", "validate_output"]

    def test_validation_errors_propagate(self) -> None:
        """Validation exceptions bubble up to caller when hooks called manually."""

        # Arrange: Plugin that raises validation error
        class ValidatingPlugin(PluginComputeBase):
            def validate_input(self, input_data: dict) -> None:
                if "required_field" not in input_data:
                    raise ValueError("Missing required_field")

            def execute(self, input_data: dict, context: dict) -> dict:
                return input_data

        plugin = ValidatingPlugin()
        input_data = {"invalid": "data"}

        # Act & Assert: Validation error propagates when called manually
        with pytest.raises(ValueError, match="Missing required_field"):
            plugin.validate_input(input_data)


class TestContextPropagation:
    """Test that context is properly propagated through execution."""

    def test_context_available_in_execute(self) -> None:
        """Context parameter is available in execute() method."""
        # Arrange
        context_received = {}

        class ContextTrackingPlugin(PluginComputeBase):
            def execute(self, input_data: dict, context: dict) -> dict:
                context_received.update(context)
                return {"correlation_id": context.get("correlation_id")}

        plugin = ContextTrackingPlugin()
        test_context = {"correlation_id": "test-789", "timestamp": "2025-01-01"}

        # Act
        result = plugin.execute({}, test_context)

        # Assert: Context was received
        assert context_received["correlation_id"] == "test-789"
        assert context_received["timestamp"] == "2025-01-01"
        assert result["correlation_id"] == "test-789"

    def test_context_available_in_validation_hooks(self) -> None:
        """Validation hooks can access context if implemented to accept it."""
        # Arrange
        # NOTE: Base class validation hooks don't receive context parameter
        # This test demonstrates that plugins CAN track context if needed

        class ContextAwarePlugin(PluginComputeBase):
            def __init__(self) -> None:
                self.last_context: dict | None = None

            def execute(self, input_data: dict, context: dict) -> dict:
                # Store context for validation hooks to access if needed
                self.last_context = context
                return {"result": "ok"}

            def validate_input(self, input_data: dict) -> None:
                # Can access context through instance variable if needed
                pass

            def validate_output(self, output_data: dict) -> None:
                # Can access context through instance variable if needed
                pass

        plugin = ContextAwarePlugin()
        test_context = {"correlation_id": "test-456"}

        # Act
        plugin.execute({}, test_context)

        # Assert: Plugin can track context internally
        assert plugin.last_context is not None
        assert plugin.last_context["correlation_id"] == "test-456"

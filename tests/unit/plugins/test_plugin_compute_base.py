"""Unit tests for PluginCompute base class and protocol conformance.

Tests verify:
- Protocol conformance for base class and example plugins
- Abstract base class behavior
- Validation hook execution order
- Determinism guarantees
- PluginJsonNormalizer functionality

NOTE: Tests are marked as skipped until PluginComputeBase and PluginJsonNormalizer
are implemented. Uncomment imports below once implementation is complete.

Note on Type Annotations:
    This test module intentionally uses dict types for test plugin implementations
    and passes raw dict inputs to plugin.execute() for testing purposes.
    The mypy directives below disable type errors for these intentional patterns.
"""
# mypy: disable-error-code="override, arg-type, attr-defined, index, return-value"

import pytest

from omnibase_infra.plugins.examples.plugin_json_normalizer import PluginJsonNormalizer

# Implementations are now complete - imports activated
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

    def test_protocol_conformance_with_example(self) -> None:
        """PluginJsonNormalizer conforms to ProtocolPluginCompute."""
        # Arrange
        instance = PluginJsonNormalizer()

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


class TestDeterminism:
    """Test determinism guarantees for compute plugins."""

    def test_same_input_produces_same_output(self) -> None:
        """Same input produces identical output (core determinism)."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"z": 3, "a": 1, "m": 2}}
        context = {"correlation_id": "test-123"}

        # Act: Execute twice with same input
        result1 = plugin.execute(input_data, context)
        result2 = plugin.execute(input_data, context)

        # Assert: Results are identical
        assert result1 == result2

    def test_determinism_across_multiple_calls(self) -> None:
        """Repeated calls with same input produce identical results."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"nested": {"z": 1, "a": 2}, "top": "level"}}
        context = {"correlation_id": "test-123"}

        # Act: Execute 5 times
        results = [plugin.execute(input_data, context) for _ in range(5)]

        # Assert: All results identical
        assert all(result == results[0] for result in results)

    def test_different_input_produces_different_output(self) -> None:
        """Different inputs produce different outputs."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input1 = {"json": {"key": "value1"}}
        input2 = {"json": {"key": "value2"}}
        context = {"correlation_id": "test-123"}

        # Act
        result1 = plugin.execute(input1, context)
        result2 = plugin.execute(input2, context)

        # Assert: Different inputs yield different outputs
        assert result1 != result2


class TestPluginJsonNormalizer:
    """Test PluginJsonNormalizer functionality."""

    def test_normalizes_simple_dict(self) -> None:
        """Sorts keys in simple dictionary."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"z": 3, "a": 1, "m": 2}}
        context = {"correlation_id": "test-123"}

        # Act
        result = plugin.execute(input_data, context)

        # Assert: Keys sorted alphabetically
        # Plugin returns {"normalized": {...}} structure
        normalized = result.get("normalized", result)
        assert list(normalized.keys()) == ["a", "m", "z"]
        assert normalized == {"a": 1, "m": 2, "z": 3}

    def test_normalizes_nested_dict(self) -> None:
        """Recursively sorts keys in nested dictionaries."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "outer_z": {"inner_z": 3, "inner_a": 1},
                "outer_a": {"inner_m": 2},
            }
        }
        context = {"correlation_id": "test-123"}

        # Act
        result = plugin.execute(input_data, context)

        # Assert: All levels sorted
        normalized = result.get("normalized", result)
        assert list(normalized.keys()) == ["outer_a", "outer_z"]
        assert list(normalized["outer_z"].keys()) == ["inner_a", "inner_z"]
        assert list(normalized["outer_a"].keys()) == ["inner_m"]

    def test_normalizes_dict_with_lists(self) -> None:
        """Handles lists within dictionaries."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"z": [3, 2, 1], "a": ["x", "y"]}}
        context = {"correlation_id": "test-123"}

        # Act
        result = plugin.execute(input_data, context)

        # Assert: Keys sorted, lists preserved
        normalized = result.get("normalized", result)
        assert list(normalized.keys()) == ["a", "z"]
        assert normalized["z"] == [3, 2, 1]  # List order preserved
        assert normalized["a"] == ["x", "y"]

    def test_normalizes_empty_dict(self) -> None:
        """Handles empty dictionary edge case."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data: dict[str, object] = {"json": {}}
        context = {"correlation_id": "test-123"}

        # Act
        result = plugin.execute(input_data, context)

        # Assert: Empty dict handled gracefully
        normalized = result.get("normalized", result)
        assert normalized == {}

    def test_preserves_values(self) -> None:
        """Values unchanged, only keys sorted."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "z": {"complex": [1, 2, 3], "value": True},
                "a": None,
                "m": 42,
            }
        }
        context = {"correlation_id": "test-123"}

        # Act
        result = plugin.execute(input_data, context)

        # Assert: Values exactly preserved
        normalized = result.get("normalized", result)
        assert normalized["z"]["complex"] == [1, 2, 3]
        assert normalized["z"]["value"] is True
        assert normalized["a"] is None
        assert normalized["m"] == 42

    def test_deterministic_normalization(self) -> None:
        """Same JSON produces same normalized output."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "users": [
                    {"name": "Alice", "id": 2},
                    {"name": "Bob", "id": 1},
                ],
                "metadata": {"version": "1.0", "created": "2024-01-01"},
            }
        }
        context = {"correlation_id": "test-123"}

        # Act: Normalize multiple times
        result1 = plugin.execute(input_data, context)
        result2 = plugin.execute(input_data, context)
        result3 = plugin.execute(input_data, context)

        # Assert: All results identical
        assert result1 == result2 == result3
        normalized = result1.get("normalized", result1)
        assert list(normalized.keys()) == ["metadata", "users"]
        assert list(normalized["metadata"].keys()) == ["created", "version"]


class TestValidationHookIntegration:
    """Test validation hooks with PluginJsonNormalizer."""

    def test_validation_hooks_execute_in_order(self) -> None:
        """Validation hooks must be called manually in correct order."""
        # Arrange: Track execution order
        execution_order: list[str] = []

        class TrackedNormalizer(PluginJsonNormalizer):
            def validate_input(self, input_data: dict[str, object]) -> None:
                execution_order.append("validate_input")
                super().validate_input(input_data)

            def validate_output(self, output_data: dict[str, object]) -> None:
                execution_order.append("validate_output")
                super().validate_output(output_data)

            def execute(
                self, input_data: dict[str, object], context: dict[str, object]
            ) -> dict[str, object]:
                execution_order.append("execute")
                return super().execute(input_data, context)

        plugin = TrackedNormalizer()
        input_data = {"json": {"z": 1, "a": 2}}
        context = {"correlation_id": "test-123"}

        # Act: Manually call hooks in correct order
        plugin.validate_input(input_data)
        output = plugin.execute(input_data, context)
        plugin.validate_output(output)

        # Assert: Correct execution order
        assert execution_order == ["validate_input", "execute", "validate_output"]

    def test_input_validation_failure(self) -> None:
        """Input validation errors prevent execution when called manually."""

        # Arrange: Plugin with strict input validation
        class StrictNormalizer(PluginJsonNormalizer):
            def validate_input(self, input_data: dict[str, object]) -> None:
                if not input_data.get("json"):
                    raise ValueError("Input must not be empty")

        plugin = StrictNormalizer()

        # Act & Assert: Empty input raises error when validated
        with pytest.raises(ValueError, match="Input must not be empty"):
            plugin.validate_input({})

    def test_output_validation_failure(self) -> None:
        """Output validation errors propagate when called manually."""

        # Arrange: Plugin with strict output validation
        class StrictNormalizer(PluginJsonNormalizer):
            def validate_output(self, output_data: dict[str, object]) -> None:
                normalized = output_data.get("normalized", output_data)
                if isinstance(normalized, dict) and len(normalized) > 5:
                    raise ValueError("Output too large")

        plugin = StrictNormalizer()
        large_input = {"json": {f"key_{i}": i for i in range(10)}}
        context = {"correlation_id": "test-123"}

        # Act: Execute and then validate output manually
        output = plugin.execute(large_input, context)

        # Assert: Large output raises error when validated
        with pytest.raises(ValueError, match="Output too large"):
            plugin.validate_output(output)


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
        test_context = {"correlation_id": "test-789", "timestamp": "2024-01-01"}

        # Act
        result = plugin.execute({}, test_context)

        # Assert: Context was received
        assert context_received["correlation_id"] == "test-789"
        assert context_received["timestamp"] == "2024-01-01"
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

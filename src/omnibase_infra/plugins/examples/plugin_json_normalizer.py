"""JSON normalization plugin for deterministic comparison.

This plugin recursively sorts JSON object keys to enable consistent comparison
and hashing. It demonstrates a pure, deterministic compute plugin with no side effects.
"""

from typing import Any, TypedDict, cast

from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase
from omnibase_infra.protocols.protocol_plugin_compute import (
    PluginContext,
    PluginInputData,
    PluginOutputData,
)

# JSON-compatible type alias for improved readability
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class JsonNormalizerInput(TypedDict, total=False):
    """Type-safe input structure for JSON normalizer.

    Fields:
        json: The JSON-compatible data structure to normalize.
    """

    json: JsonValue


class JsonNormalizerOutput(TypedDict):
    """Type-safe output structure for JSON normalizer.

    Fields:
        normalized: The normalized JSON structure with recursively sorted keys.
    """

    normalized: JsonValue


class PluginJsonNormalizer(PluginComputeBase):
    """Normalizes JSON structures for deterministic comparison."""

    def execute(
        self, input_data: PluginInputData, context: PluginContext
    ) -> PluginOutputData:
        """Execute JSON normalization with type-safe inputs and outputs."""
        json_data = cast(JsonValue, input_data.get("json", {}))
        normalized: JsonValue = self._sort_keys_recursively(json_data)
        output: JsonNormalizerOutput = {"normalized": normalized}
        return output

    def _sort_keys_recursively(self, obj: JsonValue) -> JsonValue:
        """Recursively sort dictionary keys with optimized performance.

        Performance Characteristics:
            - Time Complexity: O(n * k log k) where n is total nodes, k is keys per dict
            - Space Complexity: O(d) where d is maximum depth (recursion stack)
            - Optimizations:
              * Early type checking for primitives (most common case)
              * Sorted key iteration for dicts (single pass over items)
              * No redundant operations or object creation

        Large Structure Performance:
            For JSON with 1000+ keys, this implementation:
            - Minimizes object creation overhead
            - Uses sorted() efficiently (Timsort O(n log k))
            - Avoids redundant type checks
            - Maintains deterministic behavior

        Args:
            obj: JSON-compatible object (dict, list, or primitive)

        Returns:
            Object with recursively sorted keys (if dict), or original value

        Note:
            - Dicts: Sorted by key name (alphabetically)
            - Lists: Items processed recursively, order preserved
            - Primitives: Returned unchanged (early exit for performance)
        """
        # Early exit for primitives (most common case in large structures)
        # This optimization avoids isinstance checks for dict/list on every primitive
        if not isinstance(obj, (dict, list)):
            return obj

        if isinstance(obj, dict):
            return {k: self._sort_keys_recursively(v) for k, v in sorted(obj.items())}

        # Must be a list at this point
        return [self._sort_keys_recursively(item) for item in obj]

    def validate_input(self, input_data: PluginInputData) -> None:
        """Validate input with runtime type checking and type guards."""
        if not isinstance(input_data, dict):
            raise TypeError(f"input_data must be dict, got {type(input_data).__name__}")

        if "json" not in input_data:
            return

        json_data = input_data["json"]
        if not self._is_json_compatible(json_data):
            raise ValueError(
                f"Input 'json' must be JSON-compatible type, got {type(json_data).__name__}"
            )
        self._validate_json_structure(json_data)

    def _is_json_compatible(self, value: Any) -> bool:
        """Type guard to check if value is JSON-compatible."""
        return isinstance(value, (dict, list, str, int, float, bool, type(None)))

    def _validate_json_structure(self, obj: JsonValue) -> None:
        """Recursively validate JSON structure for non-JSON-compatible types."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not self._is_json_compatible(value):
                    raise ValueError(
                        f"Non-JSON-compatible value in dict at key '{key}': {type(value).__name__}"
                    )
                self._validate_json_structure(value)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                if not self._is_json_compatible(item):
                    raise ValueError(
                        f"Non-JSON-compatible value in list at index {index}: {type(item).__name__}"
                    )
                self._validate_json_structure(item)

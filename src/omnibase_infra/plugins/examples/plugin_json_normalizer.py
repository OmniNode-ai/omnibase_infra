"""JSON normalization plugin for deterministic comparison.

This plugin recursively sorts JSON object keys to enable consistent comparison
and hashing. It demonstrates a pure, deterministic compute plugin with no side effects.
"""

from typing import TypedDict, cast

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.errors import OnexError

from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase
from omnibase_infra.protocols.protocol_plugin_compute import (
    PluginContext,
    PluginInputData,
    PluginOutputData,
)

# JSON-compatible type alias using forward reference for recursion
# Note: This is the standard way to define recursive JSON types in Python
JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None


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
    """Normalizes JSON structures for deterministic comparison.

    Attributes:
        MAX_RECURSION_DEPTH: Maximum allowed nesting depth for JSON structures.
            Defaults to 100 levels. Override in subclass if needed.
    """

    __slots__ = ()  # Enforce statelessness - no instance attributes

    MAX_RECURSION_DEPTH: int = 100

    def execute(
        self, input_data: PluginInputData, context: PluginContext
    ) -> PluginOutputData:
        """Execute JSON normalization with type-safe inputs and outputs.

        Args:
            input_data: Dictionary containing "json" key with data to normalize
            context: Execution context (correlation_id, timestamps, etc.)

        Returns:
            Dictionary with "normalized" key containing sorted JSON

        Raises:
            OnexError: For all computation failures (with proper error chaining)
        """
        correlation_id = context.get("correlation_id")

        try:
            json_data = cast(JsonValue, input_data.get("json", {}))
            normalized: JsonValue = self._sort_keys_recursively(json_data)
            output: JsonNormalizerOutput = {"normalized": normalized}
            return output

        except RecursionError as e:
            raise OnexError(
                message="JSON structure too deeply nested for normalization",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                correlation_id=correlation_id,
                plugin_name=self.__class__.__name__,
                max_recursion_depth=self.MAX_RECURSION_DEPTH,
            ) from e

        except Exception as e:
            raise OnexError(
                message=f"Unexpected error during JSON normalization: {e}",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                correlation_id=correlation_id,
                plugin_name=self.__class__.__name__,
                input_keys=list(input_data.keys())
                if isinstance(input_data, dict)
                else [],
            ) from e

    def _sort_keys_recursively(self, obj: JsonValue, _depth: int = 0) -> JsonValue:
        """Recursively sort dictionary keys with optimized performance and depth protection.

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

        Depth Protection:
            To prevent stack overflow on deeply nested structures, recursion depth
            is limited to MAX_RECURSION_DEPTH (default: 100 levels). This protects
            against maliciously crafted or malformed JSON that could exhaust the
            Python call stack.

        Args:
            obj: JSON-compatible object (dict, list, or primitive)
            _depth: Internal depth counter for recursion protection. Do not set
                manually; this is tracked automatically during recursion.

        Returns:
            Object with recursively sorted keys (if dict), or original value

        Raises:
            RecursionError: If nesting depth exceeds MAX_RECURSION_DEPTH levels.

        Note:
            - Dicts: Sorted by key name (alphabetically)
            - Lists: Items processed recursively, order preserved
            - Primitives: Returned unchanged (early exit for performance)
        """
        # Depth protection to prevent stack overflow on deeply nested structures
        if _depth > self.MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"JSON structure exceeds maximum nesting depth of "
                f"{self.MAX_RECURSION_DEPTH} levels"
            )

        # Early exit for primitives (most common case in large structures)
        # This optimization avoids isinstance checks for dict/list on every primitive
        if not isinstance(obj, (dict, list)):
            return obj

        if isinstance(obj, dict):
            return {
                k: self._sort_keys_recursively(v, _depth + 1)
                for k, v in sorted(obj.items())
            }

        # Must be a list at this point
        return [self._sort_keys_recursively(item, _depth + 1) for item in obj]

    def validate_input(self, input_data: PluginInputData) -> None:
        """Validate input with runtime type checking and type guards.

        Args:
            input_data: The input data to validate

        Raises:
            TypeError: If input_data is not a dict
            ValueError: If "json" key exists but is not JSON-compatible type.
                Caller should wrap in OnexError with correlation_id.
            OnexError: If JSON structure exceeds maximum nesting depth
        """
        if not isinstance(input_data, dict):
            raise TypeError(f"input_data must be dict, got {type(input_data).__name__}")

        if "json" not in input_data:
            return

        json_data = input_data["json"]
        if not self._is_json_compatible(json_data):
            raise ValueError(
                f"Input 'json' must be JSON-compatible type, got {type(json_data).__name__}"
            )

        try:
            self._validate_json_structure(json_data)
        except RecursionError as e:
            raise OnexError(
                message="JSON structure too deeply nested for validation",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                plugin_name=self.__class__.__name__,
                max_recursion_depth=self.MAX_RECURSION_DEPTH,
            ) from e

    def _is_json_compatible(self, value: object) -> bool:
        """Type guard to check if value is JSON-compatible."""
        return isinstance(value, (dict, list, str, int, float, bool, type(None)))

    def _validate_json_structure(self, obj: JsonValue, _depth: int = 0) -> None:
        """Recursively validate JSON structure for non-JSON-compatible types.

        Args:
            obj: JSON-compatible object to validate
            _depth: Internal depth counter for recursion protection. Do not set
                manually; this is tracked automatically during recursion.

        Raises:
            ValueError: If non-JSON-compatible types are found
            RecursionError: If nesting depth exceeds MAX_RECURSION_DEPTH levels
        """
        # Depth protection to prevent stack overflow during validation
        if _depth > self.MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"JSON structure exceeds maximum nesting depth of "
                f"{self.MAX_RECURSION_DEPTH} levels"
            )

        if isinstance(obj, dict):
            for key, value in obj.items():
                if not self._is_json_compatible(value):
                    raise ValueError(
                        f"Non-JSON-compatible value in dict at key '{key}': {type(value).__name__}"
                    )
                self._validate_json_structure(value, _depth + 1)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                if not self._is_json_compatible(item):
                    raise ValueError(
                        f"Non-JSON-compatible value in list at index {index}: {type(item).__name__}"
                    )
                self._validate_json_structure(item, _depth + 1)

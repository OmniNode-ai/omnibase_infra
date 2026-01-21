"""JSON normalization plugin for deterministic comparison.

This plugin recursively sorts JSON object keys to enable consistent comparison
and hashing. It demonstrates a pure, deterministic compute plugin with no side effects.
"""

from typing import TypedDict, cast

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.errors import OnexError
from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase
from omnibase_infra.protocols.protocol_plugin_compute import (
    ModelPluginContext,
    ModelPluginInputData,
    ModelPluginOutputData,
)


class JsonNormalizerInput(TypedDict, total=False):
    """Type-safe input structure for JSON normalizer.

    Fields:
        json: The JSON-compatible data structure to normalize.
    """

    json: object


class JsonNormalizerOutput(TypedDict):
    """Type-safe output structure for JSON normalizer.

    Fields:
        normalized: The normalized JSON structure with recursively sorted keys.
    """

    normalized: object


class PluginJsonNormalizer(PluginComputeBase):
    """Normalizes JSON structures for deterministic comparison.

    Attributes:
        MAX_RECURSION_DEPTH: Maximum allowed nesting depth for JSON structures.
            Defaults to 100 levels. Override in subclass if needed.
    """

    __slots__ = ()  # Enforce statelessness - no instance attributes

    MAX_RECURSION_DEPTH: int = 100

    def execute(
        self, input_data: ModelPluginInputData, context: ModelPluginContext
    ) -> ModelPluginOutputData:
        """Execute JSON normalization with type-safe inputs and outputs.

        Args:
            input_data: Dictionary containing "json" key with data to normalize
            context: Execution context containing:
                - correlation_id: For error tracing
                - max_recursion_depth: Optional override for MAX_RECURSION_DEPTH

        Returns:
            Dictionary with "normalized" key containing sorted JSON

        Raises:
            OnexError: For all computation failures (with proper error chaining)
        """
        correlation_id = context.get("correlation_id")

        # Allow context to override max recursion depth for specific use cases
        max_depth_override = context.get("max_recursion_depth")
        if max_depth_override is not None and isinstance(max_depth_override, int):
            effective_max_depth = max_depth_override
        else:
            effective_max_depth = self.MAX_RECURSION_DEPTH

        try:
            json_data = cast("object", input_data.get("json", {}))
            normalized: object = self._sort_keys_recursively(
                json_data, _max_depth=effective_max_depth
            )
            output: JsonNormalizerOutput = {"normalized": normalized}
            return cast("ModelPluginOutputData", output)

        except RecursionError as e:
            raise OnexError(
                message="JSON structure too deeply nested for normalization",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                correlation_id=correlation_id,
                plugin_name=self.__class__.__name__,
                max_recursion_depth=effective_max_depth,
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

    def _sort_keys_recursively(
        self, obj: object, _depth: int = 0, _max_depth: int | None = None
    ) -> object:
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
            is limited to _max_depth (default: MAX_RECURSION_DEPTH = 100 levels).
            This protects against maliciously crafted or malformed JSON that could
            exhaust the Python call stack.

        Args:
            obj: JSON-compatible object (dict, list, or primitive)
            _depth: Internal depth counter for recursion protection. Do not set
                manually; this is tracked automatically during recursion.
            _max_depth: Maximum allowed recursion depth. If None, uses
                MAX_RECURSION_DEPTH class attribute. Can be overridden via
                context["max_recursion_depth"] in execute().

        Returns:
            Object with recursively sorted keys (if dict), or original value

        Raises:
            RecursionError: If nesting depth exceeds maximum allowed depth.

        Note:
            - Dicts: Sorted by key name (alphabetically)
            - Lists: Items processed recursively, order preserved
            - Primitives: Returned unchanged (early exit for performance)
        """
        effective_max = (
            _max_depth if _max_depth is not None else self.MAX_RECURSION_DEPTH
        )

        # Depth protection to prevent stack overflow on deeply nested structures
        if _depth > effective_max:
            raise RecursionError(
                f"JSON structure exceeds maximum nesting depth of "
                f"{effective_max} levels"
            )

        # Early exit for primitives (most common case in large structures)
        # This optimization avoids isinstance checks for dict/list on every primitive
        if not isinstance(obj, dict | list):
            return obj

        if isinstance(obj, dict):
            return {
                k: self._sort_keys_recursively(v, _depth + 1, _max_depth=effective_max)
                for k, v in sorted(obj.items())
            }

        # Must be a list at this point
        return [
            self._sort_keys_recursively(item, _depth + 1, _max_depth=effective_max)
            for item in obj
        ]

    def validate_input(self, input_data: ModelPluginInputData) -> None:
        """Validate input with runtime type checking and type guards.

        This method validates input structure before execution. By protocol design,
        context is NOT available during validation - only the raw input_data.

        Error Handling Contract:
            This method raises TypeError/ValueError for validation failures.
            These are intentionally NOT wrapped in OnexError here because:
            1. Context (with correlation_id) is not available in validate_input
            2. The caller (registry/executor) wraps these in OnexError with context
            3. This follows the ONEX plugin validation hook pattern

            For example, the plugin executor wraps validation errors:
            ```
            try:
                plugin.validate_input(input_data)
            except (TypeError, ValueError) as e:
                raise OnexError(
                    message=str(e),
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    correlation_id=context.get("correlation_id"),
                ) from e
            ```

        Args:
            input_data: The input data to validate (context not available here)

        Raises:
            TypeError: If input_data is not a dict. Caller wraps in OnexError.
            ValueError: If "json" key exists but is not JSON-compatible type.
                Caller wraps in OnexError with correlation_id from context.
            OnexError: If JSON structure exceeds maximum nesting depth (direct
                raise since this is an internal error, not a validation error).
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
        return isinstance(value, dict | list | str | int | float | bool | type(None))

    def _validate_json_structure(self, obj: object, _depth: int = 0) -> None:
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

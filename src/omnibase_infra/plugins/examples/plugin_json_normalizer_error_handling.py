"""JSON normalization plugin demonstrating ONEX error handling standards.

This plugin recursively sorts JSON object keys to enable consistent comparison
and hashing. It demonstrates a pure, deterministic compute plugin with comprehensive
error handling following ONEX standards.

Error Handling:
    This plugin demonstrates ONEX error handling standards:
    - OnexError chaining for all exceptions
    - Correlation ID propagation from context
    - Never suppress errors silently
    - Preserve full error context for debugging

Example Error Scenarios:
    1. Invalid input type (non-JSON-compatible)
    2. Deeply nested structures causing recursion errors
    3. Malformed input data structures
"""

from typing import cast

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.errors import OnexError
from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase
from omnibase_infra.protocols.protocol_plugin_compute import (
    ModelPluginContext,
    ModelPluginInputData,
    ModelPluginOutputData,
)


class PluginJsonNormalizerErrorHandling(PluginComputeBase):
    """JSON normalizer with comprehensive ONEX error handling.

    This is an enhanced version of PluginJsonNormalizer that demonstrates
    proper error handling patterns following ONEX standards.
    """

    __slots__ = ()  # Enforce statelessness - no instance attributes

    def execute(
        self, input_data: ModelPluginInputData, context: ModelPluginContext
    ) -> ModelPluginOutputData:
        """Execute JSON normalization with comprehensive error handling.

        Args:
            input_data: Dictionary containing "json" key with data to normalize
            context: Execution context (correlation_id, timestamps, etc.)

        Returns:
            Dictionary with "normalized" key containing sorted JSON

        Raises:
            OnexError: For all computation failures (with proper error chaining)

        Error Handling Examples:
            ```python
            from omnibase_core.errors import OnexError
            from uuid import uuid4

            # Example 1: Recursion depth error
            try:
                deeply_nested = create_deeply_nested_dict(depth=1500)
                plugin.execute({"json": deeply_nested}, {"correlation_id": uuid4()})
            except OnexError as e:
                assert e.model.error_code == EnumCoreErrorCode.INTERNAL_ERROR
                assert e.model.correlation_id is not None
                assert e.model.details["max_recursion_depth"] == 1000

            # Example 2: Unexpected error
            try:
                plugin.execute({"json": valid_data}, {"correlation_id": uuid4()})
            except OnexError as e:
                assert e.model.error_code == EnumCoreErrorCode.INTERNAL_ERROR
                assert "input_keys" in e.model.details
            ```
        """
        # Extract correlation_id for error handling and tracing
        correlation_id = context.get("correlation_id", "unknown")

        try:
            # Retrieve JSON data with safe default
            json_data = cast("object", input_data.get("json", {}))

            # Perform pure deterministic computation
            normalized = self._sort_keys_recursively(json_data)

            # Return result with correlation_id for tracing
            return cast(
                "ModelPluginOutputData",
                {
                    "normalized": normalized,
                    "correlation_id": correlation_id,
                },
            )

        except RecursionError as e:
            # Handle deeply nested structures exceeding Python's recursion limit
            raise OnexError(
                message="JSON structure too deeply nested for normalization",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                correlation_id=correlation_id,
                plugin_name=self.__class__.__name__,
                max_recursion_depth=1000,  # Python's default recursion limit
                input_depth_estimate="exceeded",
            ) from e

        except TypeError as e:
            # Handle type errors during sorting (e.g., comparing incompatible types)
            raise OnexError(
                message=f"Type error during JSON normalization: {e}",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                correlation_id=correlation_id,
                plugin_name=self.__class__.__name__,
                input_keys=list(input_data.model_dump().keys()),
            ) from e

        except Exception as e:
            # Catch-all for unexpected errors (should not happen in pure computation)
            raise OnexError(
                message=f"Unexpected error during JSON normalization: {e}",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                correlation_id=correlation_id,
                plugin_name=self.__class__.__name__,
                input_keys=list(input_data.model_dump().keys()),
                exception_type=type(e).__name__,
            ) from e

    def _sort_keys_recursively(self, obj: object) -> object:
        """Recursively sort dictionary keys.

        Args:
            obj: JSON-compatible object (dict, list, or primitive)

        Returns:
            Object with recursively sorted keys (if dict), or original value

        Raises:
            RecursionError: If nesting exceeds Python's recursion limit
            TypeError: If comparing incompatible types during sorting

        Note:
            - Dicts: Sorted by key name (alphabetically)
            - Lists: Items processed recursively, order preserved
            - Primitives: Returned unchanged
            - This method does not include explicit depth protection unlike
              the main PluginJsonNormalizer. It relies on Python's default
              recursion limit for simplicity in this error handling example.
        """
        if isinstance(obj, dict):
            # Sort keys and recursively process values
            # May raise TypeError if keys are incomparable
            return {k: self._sort_keys_recursively(v) for k, v in sorted(obj.items())}

        if isinstance(obj, list):
            # Recursively process list items, preserve order
            # May raise RecursionError if deeply nested
            return [self._sort_keys_recursively(item) for item in obj]

        # Primitive values (str, int, float, bool, None) returned unchanged
        return obj

    def validate_input(self, input_data: ModelPluginInputData) -> None:
        """Validate that input contains JSON-compatible data.

        Args:
            input_data: The input data to validate

        Raises:
            ValueError: If "json" key exists but is not JSON-compatible type.
                Caller should wrap in OnexError with correlation_id.

        Error Handling:
            This method raises ValueError for invalid input types. The caller
            (node/executor) is responsible for catching and wrapping in OnexError.

        Example Caller Wrapping:
            ```python
            from omnibase_core.errors import OnexError
            from omnibase_core.enums import EnumCoreErrorCode

            correlation_id = context.get("correlation_id", "unknown")

            try:
                plugin.validate_input(input_data)
            except ValueError as e:
                raise OnexError(
                    message=f"Input validation failed: {e}",
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    correlation_id=correlation_id,
                    plugin_name="PluginJsonNormalizerErrorHandling",
                    expected_types=["dict", "list", "str", "int", "float", "bool", "None"],
                    actual_type=type(input_data.get("json")).__name__,
                ) from e
            ```

        Note:
            Missing "json" key is valid - plugin returns empty normalized dict.
        """
        json_data = input_data.get("json")
        if json_data is not None:
            # Ensure it's a JSON-compatible type
            if not isinstance(
                json_data, dict | list | str | int | float | bool | type(None)
            ):
                raise ValueError(
                    f"Input 'json' must be JSON-compatible type "
                    f"(dict, list, str, int, float, bool, None), "
                    f"got {type(json_data).__name__}"
                )

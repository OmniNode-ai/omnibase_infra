"""JSON normalization plugin for deterministic comparison.

This plugin recursively sorts JSON object keys to enable consistent comparison
and hashing. It demonstrates a pure, deterministic compute plugin with no side effects.
"""

from typing import Any

from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase


class PluginJsonNormalizer(PluginComputeBase):
    """Normalizes JSON structures for deterministic comparison.

    This plugin performs deep key sorting on JSON-compatible data structures,
    ensuring that two semantically identical objects with different key orders
    produce identical normalized output.

    Determinism Guarantee:
        Given the same input JSON structure, this plugin ALWAYS produces the
        same normalized output. Key sorting is deterministic, recursive, and
        has no external dependencies.

    Use Cases:
        - JSON comparison and diffing
        - Deterministic hashing of JSON objects
        - Canonical JSON representation for signatures
        - Test fixtures and snapshots

    Example:
        ```python
        plugin = PluginJsonNormalizer()

        # Input with arbitrary key order
        input_data = {
            "json": {
                "z": 3,
                "a": 1,
                "m": {"nested": "value", "another": "key"}
            }
        }

        # Output with deterministic key order
        result = plugin.execute(input_data, context={})
        # result["normalized"] == {
        #     "a": 1,
        #     "m": {"another": "key", "nested": "value"},
        #     "z": 3
        # }
        ```

    Args:
        input_data: Dictionary containing "json" key with JSON-compatible data
        context: Execution context (unused but required by protocol)

    Returns:
        Dictionary with "normalized" key containing sorted JSON structure

    Note:
        - Only processes data under the "json" key in input_data
        - Returns empty dict if "json" key is missing
        - Handles nested dicts, lists, and primitive values
        - Preserves all values, only reorders keys
    """

    def execute(
        self, input_data: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute JSON normalization.

        Args:
            input_data: Dictionary containing "json" key with data to normalize
            context: Execution context (correlation_id, timestamps, etc.)

        Returns:
            Dictionary with "normalized" key containing sorted JSON

        Example:
            >>> plugin = PluginJsonNormalizer()
            >>> result = plugin.execute({"json": {"b": 2, "a": 1}}, {})
            >>> result["normalized"]
            {"a": 1, "b": 2}
        """
        json_data = input_data.get("json", {})
        normalized = self._sort_keys_recursively(json_data)
        return {"normalized": normalized}

    def _sort_keys_recursively(self, obj: Any) -> Any:
        """Recursively sort dictionary keys.

        Args:
            obj: JSON-compatible object (dict, list, or primitive)

        Returns:
            Object with recursively sorted keys (if dict), or original value

        Note:
            - Dicts: Sorted by key name (alphabetically)
            - Lists: Items processed recursively, order preserved
            - Primitives: Returned unchanged
        """
        if isinstance(obj, dict):
            # Sort keys and recursively process values
            return {k: self._sort_keys_recursively(v) for k, v in sorted(obj.items())}

        if isinstance(obj, list):
            # Recursively process list items, preserve order
            return [self._sort_keys_recursively(item) for item in obj]

        # Primitive values (str, int, float, bool, None) returned unchanged
        return obj

    def validate_input(self, input_data: dict[str, Any]) -> None:
        """Validate that input contains JSON-compatible data.

        Args:
            input_data: The input data to validate

        Raises:
            ValueError: If "json" key exists but is not dict/list/primitive

        Note:
            Missing "json" key is valid - plugin returns empty normalized dict
        """
        if "json" in input_data:
            json_data = input_data["json"]
            # Ensure it's a JSON-compatible type
            if not isinstance(
                json_data, (dict, list, str, int, float, bool, type(None))
            ):
                raise ValueError(
                    f"Input 'json' must be JSON-compatible type, got {type(json_data).__name__}"
                )

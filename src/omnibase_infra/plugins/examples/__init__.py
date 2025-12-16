"""Example compute plugins demonstrating deterministic computation patterns.

This module contains reference implementations that demonstrate best practices
for compute plugin development.

Available Examples:
    - PluginJsonNormalizer - Deterministic JSON key sorting for comparison

Usage:
    ```python
    from omnibase_infra.plugins.examples import PluginJsonNormalizer

    plugin = PluginJsonNormalizer()
    result = plugin.execute(
        input_data={"json": {"z": 3, "a": 1}},
        context={"correlation_id": "123"}
    )
    # result["normalized"] == {"a": 1, "z": 3}
    ```
"""

from omnibase_infra.plugins.examples.plugin_json_normalizer import (
    PluginJsonNormalizer,
)

__all__ = [
    "PluginJsonNormalizer",
]

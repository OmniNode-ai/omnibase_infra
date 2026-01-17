"""Tests for PluginJsonNormalizer with performance validation.

Note on Type Annotations:
    This test module intentionally passes raw dict types to plugin.execute() instead of
    ModelPluginInputData/ModelPluginContext to test the plugin's behavior with raw dict
    inputs. The mypy directive below disables type errors for these intentional patterns.
"""
# mypy: disable-error-code="arg-type, attr-defined, index, var-annotated"

import time

import pytest

from omnibase_core.errors import OnexError
from omnibase_infra.plugins.examples.plugin_json_normalizer import (
    PluginJsonNormalizer,
)


class TestPluginJsonNormalizer:
    """Test suite for JSON normalization plugin."""

    @pytest.fixture
    def plugin(self) -> PluginJsonNormalizer:
        """Create plugin instance for testing."""
        return PluginJsonNormalizer()

    def test_basic_key_sorting(self, plugin: PluginJsonNormalizer) -> None:
        """Test basic dictionary key sorting."""
        input_data = {"json": {"z": 3, "a": 1, "m": 2}}
        result = plugin.execute(input_data, {})

        assert result["normalized"] == {"a": 1, "m": 2, "z": 3}

    def test_nested_key_sorting(self, plugin: PluginJsonNormalizer) -> None:
        """Test recursive key sorting in nested structures."""
        input_data = {
            "json": {
                "z": 3,
                "a": 1,
                "m": {"nested": "value", "another": "key", "zebra": "last"},
            }
        }
        result = plugin.execute(input_data, {})

        expected = {
            "a": 1,
            "m": {"another": "key", "nested": "value", "zebra": "last"},
            "z": 3,
        }
        assert result["normalized"] == expected

    def test_list_preservation(self, plugin: PluginJsonNormalizer) -> None:
        """Test that list order is preserved during normalization."""
        input_data = {"json": {"items": [{"z": 1}, {"a": 2}, {"m": 3}]}}
        result = plugin.execute(input_data, {})

        # List order preserved, but dict keys sorted
        expected = {"items": [{"z": 1}, {"a": 2}, {"m": 3}]}
        assert result["normalized"] == expected

    def test_primitive_values(self, plugin: PluginJsonNormalizer) -> None:
        """Test handling of primitive values."""
        input_data = {
            "json": {
                "string": "value",
                "number": 42,
                "float": 3.14,
                "bool": True,
                "null": None,
            }
        }
        result = plugin.execute(input_data, {})

        # Keys sorted, values unchanged
        expected = {
            "bool": True,
            "float": 3.14,
            "null": None,
            "number": 42,
            "string": "value",
        }
        assert result["normalized"] == expected

    def test_empty_structures(self, plugin: PluginJsonNormalizer) -> None:
        """Test handling of empty dicts and lists."""
        input_data = {"json": {"empty_dict": {}, "empty_list": []}}
        result = plugin.execute(input_data, {})

        expected = {"empty_dict": {}, "empty_list": []}
        assert result["normalized"] == expected

    def test_missing_json_key(self, plugin: PluginJsonNormalizer) -> None:
        """Test behavior when 'json' key is missing."""
        input_data = {"other": "data"}
        result = plugin.execute(input_data, {})

        assert result["normalized"] == {}

    def test_determinism(self, plugin: PluginJsonNormalizer) -> None:
        """Test that same input produces identical output."""
        input_data = {
            "json": {
                "z": {"nested": [1, 2, 3]},
                "a": {"another": {"deep": "value"}},
            }
        }

        result1 = plugin.execute(input_data, {})
        result2 = plugin.execute(input_data, {})

        assert result1 == result2
        assert result1["normalized"] == result2["normalized"]

    def test_input_validation_valid(self, plugin: PluginJsonNormalizer) -> None:
        """Test input validation accepts valid JSON types."""
        valid_inputs = [
            {"json": {}},
            {"json": []},
            {"json": "string"},
            {"json": 42},
            {"json": 3.14},
            {"json": True},
            {"json": None},
        ]

        for input_data in valid_inputs:
            plugin.validate_input(input_data)  # Should not raise

    def test_input_validation_invalid(self, plugin: PluginJsonNormalizer) -> None:
        """Test input validation rejects invalid types."""
        invalid_input = {"json": object()}  # Not JSON-compatible

        with pytest.raises(
            ValueError, match="Input 'json' must be JSON-compatible type"
        ):
            plugin.validate_input(invalid_input)

    def test_deeply_nested_structure(self, plugin: PluginJsonNormalizer) -> None:
        """Test handling of deeply nested structures."""
        # Create 10-level deep nesting
        nested: dict[str, object] = {"level_10": "deep"}
        for i in range(9, 0, -1):
            nested = {f"level_{i}": nested}

        input_data = {"json": {"z": nested, "a": "top"}}
        result = plugin.execute(input_data, {})

        # Verify top-level keys are sorted
        keys = list(result["normalized"].keys())
        assert keys == ["a", "z"]

    def test_large_structure_performance(self, plugin: PluginJsonNormalizer) -> None:
        """Test performance optimization for large JSON structures.

        This test validates the O(n * k log k) complexity optimization
        by measuring execution time for structures with 1000+ keys.
        """
        # Create large structure: 1000 top-level keys
        large_json = {
            f"key_{i:04d}": {
                "nested_a": i,
                "nested_z": i * 2,
                "nested_m": [{"item_z": i, "item_a": i + 1}],
            }
            for i in range(1000)
        }

        input_data = {"json": large_json}

        # Measure execution time
        start_time = time.perf_counter()
        result = plugin.execute(input_data, {})
        elapsed_time = time.perf_counter() - start_time

        # Verify correctness
        assert len(result["normalized"]) == 1000

        # Verify keys are sorted
        keys = list(result["normalized"].keys())
        assert keys == sorted(keys)

        # Verify nested keys are sorted
        first_item = result["normalized"]["key_0000"]
        assert list(first_item.keys()) == ["nested_a", "nested_m", "nested_z"]

        # Performance assertion: should complete in reasonable time
        # With optimizations, 1000 keys should take < 50ms on modern hardware
        # CI environments have variable performance due to shared resources,
        # containerization overhead, and CPU throttling. Using 0.5s threshold
        # to prevent flaky failures while still catching severe regressions.
        assert elapsed_time < 0.5, (
            f"Performance regression: took {elapsed_time:.3f}s for 1000 keys"
        )

    def test_wide_structure_performance(self, plugin: PluginJsonNormalizer) -> None:
        """Test performance for structures with many keys per level.

        Validates that sorting optimization (Timsort O(n log n)) is efficient
        for wide dictionaries with hundreds of keys.
        """
        # Create structure with 500 keys at same level
        wide_json = {f"key_{i:03d}": f"value_{i}" for i in range(500)}

        input_data = {"json": wide_json}

        start_time = time.perf_counter()
        result = plugin.execute(input_data, {})
        elapsed_time = time.perf_counter() - start_time

        # Verify correctness
        assert len(result["normalized"]) == 500
        assert list(result["normalized"].keys()) == sorted(wide_json.keys())

        # Should be very fast for flat structure
        # Using CI-friendly threshold (0.25s) to account for environment variability
        assert elapsed_time < 0.25, (
            f"Performance regression: took {elapsed_time:.3f}s for 500 keys"
        )

    def test_mixed_structure_performance(self, plugin: PluginJsonNormalizer) -> None:
        """Test performance for mixed depth and width structures.

        Validates optimization for realistic JSON with varying nesting levels.
        """
        # Create realistic structure: moderate width and depth
        mixed_json = {
            f"category_{i}": {
                f"subcategory_{j}": {
                    "items": [{f"field_{k}": k for k in range(10)} for _ in range(5)]
                }
                for j in range(10)
            }
            for i in range(20)
        }

        input_data = {"json": mixed_json}

        start_time = time.perf_counter()
        result = plugin.execute(input_data, {})
        elapsed_time = time.perf_counter() - start_time

        # Verify correctness
        assert len(result["normalized"]) == 20

        # Verify nested sorting
        first_category = result["normalized"]["category_0"]
        assert list(first_category.keys()) == sorted(first_category.keys())

        # Should complete efficiently
        # Using CI-friendly threshold (0.75s) to account for environment variability
        assert elapsed_time < 0.75, (
            f"Performance regression: took {elapsed_time:.3f}s for mixed structure"
        )

    def test_primitive_heavy_structure(self, plugin: PluginJsonNormalizer) -> None:
        """Test early exit optimization for structures with many primitives.

        Validates that early type checking optimization improves performance
        when most nodes are primitives (common case).
        """
        # Create structure heavily weighted toward primitives
        primitive_heavy = {
            f"key_{i}": i if i % 2 == 0 else f"string_{i}" for i in range(1000)
        }

        input_data = {"json": primitive_heavy}

        start_time = time.perf_counter()
        result = plugin.execute(input_data, {})
        elapsed_time = time.perf_counter() - start_time

        # Verify correctness
        assert len(result["normalized"]) == 1000

        # Early exit optimization should make this very fast
        # Using CI-friendly threshold (0.25s) to account for environment variability
        assert elapsed_time < 0.25, (
            f"Early exit optimization failed: took {elapsed_time:.3f}s"
        )

    def test_recursion_depth_limit_exceeded(self, plugin: PluginJsonNormalizer) -> None:
        """Test that deeply nested structures exceeding MAX_RECURSION_DEPTH raise OnexError.

        This test validates the depth protection mechanism that prevents stack overflow
        on maliciously crafted or malformed JSON with excessive nesting.
        """
        # Create structure that exceeds the default MAX_RECURSION_DEPTH (100)
        depth = plugin.MAX_RECURSION_DEPTH + 5  # 105 levels deep
        nested: dict[str, object] = {"deepest": "value"}
        for i in range(depth - 1):
            nested = {f"level_{i}": nested}

        input_data = {"json": nested}

        with pytest.raises(OnexError) as exc_info:
            plugin.execute(input_data, {})

        # Verify error message contains depth-related information
        error_msg = str(exc_info.value).lower()
        assert "deeply nested" in error_msg or "too deeply nested" in error_msg
        # Verify the original RecursionError is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RecursionError)

    def test_recursion_depth_at_limit_succeeds(
        self, plugin: PluginJsonNormalizer
    ) -> None:
        """Test that structures exactly at MAX_RECURSION_DEPTH succeed.

        Validates that the depth check uses > (not >=) to allow structures
        exactly at the limit to be processed.
        """
        # Create structure exactly at the limit (100 levels)
        depth = plugin.MAX_RECURSION_DEPTH
        nested: dict[str, object] = {"deepest": "value"}
        for i in range(depth - 1):
            nested = {f"level_{i}": nested}

        input_data = {"json": nested}

        # Should succeed without raising RecursionError
        result = plugin.execute(input_data, {})
        assert "normalized" in result

    def test_recursion_depth_limit_with_lists(
        self, plugin: PluginJsonNormalizer
    ) -> None:
        """Test depth protection applies to list nesting as well.

        Validates that depth tracking works correctly for mixed dict/list structures.
        """
        # Create deeply nested list structure
        depth = plugin.MAX_RECURSION_DEPTH + 5
        nested: list[object] = ["deepest"]
        for _ in range(depth - 1):
            nested = [nested]

        input_data = {"json": nested}

        with pytest.raises(OnexError) as exc_info:
            plugin.execute(input_data, {})

        error_msg = str(exc_info.value).lower()
        assert "deeply nested" in error_msg or "too deeply nested" in error_msg
        assert isinstance(exc_info.value.__cause__, RecursionError)

    def test_validate_input_depth_protection(
        self, plugin: PluginJsonNormalizer
    ) -> None:
        """Test that validate_input has depth protection to prevent stack overflow.

        This ensures malicious deeply nested JSON cannot cause stack overflow
        during input validation before reaching the protected execute path.
        """
        # Create structure deeper than MAX_RECURSION_DEPTH
        depth = plugin.MAX_RECURSION_DEPTH + 5
        nested: dict[str, object] = {"deepest": "value"}
        for i in range(depth - 1):
            nested = {f"level_{i}": nested}

        input_data: dict[str, object] = {"json": nested}

        with pytest.raises(OnexError) as exc_info:
            plugin.validate_input(input_data)

        error_msg = str(exc_info.value).lower()
        assert "deeply nested" in error_msg or "too deeply nested" in error_msg
        assert isinstance(exc_info.value.__cause__, RecursionError)

    def test_validate_input_depth_protection_with_lists(
        self, plugin: PluginJsonNormalizer
    ) -> None:
        """Test validate_input depth protection for nested lists."""
        depth = plugin.MAX_RECURSION_DEPTH + 5
        nested: list[object] = ["deepest"]
        for _ in range(depth - 1):
            nested = [nested]

        input_data: dict[str, object] = {"json": nested}

        with pytest.raises(OnexError) as exc_info:
            plugin.validate_input(input_data)

        error_msg = str(exc_info.value).lower()
        assert "deeply nested" in error_msg or "too deeply nested" in error_msg
        assert isinstance(exc_info.value.__cause__, RecursionError)

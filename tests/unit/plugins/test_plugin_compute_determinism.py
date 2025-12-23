"""Comprehensive determinism verification tests for compute plugins.

This module provides extensive testing of determinism guarantees for compute plugins
following the ProtocolPluginCompute contract. All compute plugins MUST pass these tests
to ensure they satisfy the determinism requirements.

Test Categories:
- Repeatability: Same input → Same output (100+ iterations)
- Object Identity Independence: Different instances → Same result
- Concurrent Execution Safety: Parallel determinism
- Input Immutability: No modification of inputs
- State Independence: No mutable state between calls
- Call Order Independence: Results independent of execution order

Requirements:
- Tests verify bit-for-bit output equality
- Tests cover edge cases (empty inputs, nested structures, large datasets)
- Tests verify thread safety without locks affecting determinism
- Tests verify no side effects or state leakage
"""

import concurrent.futures
import copy
from typing import Any


from omnibase_infra.plugins.examples.plugin_json_normalizer import PluginJsonNormalizer


class TestRepeatabilityRequirement:
    """Test Requirement 1: Same input → Same output (100+ iterations)."""

    def test_repeatability_100_iterations_simple_input(self) -> None:
        """Same simple input produces identical output across 100 iterations."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"z": 3, "a": 1, "m": 2}}
        context = {"correlation_id": "test-repeat-simple"}

        # Act: Execute 100 times
        results = [plugin.execute(input_data, context) for _ in range(100)]

        # Assert: All results are bit-for-bit identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "REPEATABILITY VIOLATION: Same input did not produce same output"

    def test_repeatability_100_iterations_nested_input(self) -> None:
        """Same nested input produces identical output across 100 iterations."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "level1": {"level2": {"level3": {"value": 42}}},
                "another_key": "test",
            }
        }
        context = {"correlation_id": "test-repeat-nested"}

        # Act: Execute 100 times
        results = [plugin.execute(input_data, context) for _ in range(100)]

        # Assert: All results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "REPEATABILITY VIOLATION: Nested input produced non-deterministic results"

    def test_repeatability_100_iterations_large_input(self) -> None:
        """Same large input produces identical output across 100 iterations."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {f"key_{i}": f"value_{i}" for i in range(100)}}
        context = {"correlation_id": "test-repeat-large"}

        # Act: Execute 100 times
        results = [plugin.execute(input_data, context) for _ in range(100)]

        # Assert: All results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "REPEATABILITY VIOLATION: Large input produced non-deterministic results"

    def test_repeatability_with_lists(self) -> None:
        """Repeatability holds for inputs containing lists."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "items": [1, 2, 3, 4, 5],
                "nested_lists": [[1, 2], [3, 4], [5, 6]],
            }
        }
        context = {"correlation_id": "test-repeat-lists"}

        # Act: Execute 100 times
        results = [plugin.execute(input_data, context) for _ in range(100)]

        # Assert: All results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "REPEATABILITY VIOLATION: List inputs produced non-deterministic results"

    def test_repeatability_empty_input(self) -> None:
        """Repeatability holds for empty input."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data: dict[str, Any] = {"json": {}}
        context = {"correlation_id": "test-repeat-empty"}

        # Act: Execute 100 times
        results = [plugin.execute(input_data, context) for _ in range(100)]

        # Assert: All results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "REPEATABILITY VIOLATION: Empty input produced non-deterministic results"


class TestObjectIdentityIndependence:
    """Test Requirement 2: Different instances → Same result."""

    def test_different_instances_same_result(self) -> None:
        """Different plugin instances produce identical results."""
        # Arrange
        input_data = {"json": {"key1": "value1", "key2": "value2", "key3": "value3"}}
        context = {"correlation_id": "test-identity"}

        # Act: Create 10 different instances and execute
        results = [
            PluginJsonNormalizer().execute(input_data, context) for _ in range(10)
        ]

        # Assert: All instances produce identical results
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "OBJECT IDENTITY VIOLATION: Different instances produced different results"

    def test_instance_reuse_vs_fresh_instances(self) -> None:
        """Reused instance produces same result as fresh instances."""
        # Arrange
        input_data = {"json": {"test": "data"}}
        context = {"correlation_id": "test-reuse"}

        reused_plugin = PluginJsonNormalizer()

        # Act: Execute with reused instance and fresh instances
        reused_results = [reused_plugin.execute(input_data, context) for _ in range(5)]
        fresh_results = [
            PluginJsonNormalizer().execute(input_data, context) for _ in range(5)
        ]

        # Assert: Reused and fresh instances produce identical results
        assert all(
            result == reused_results[0] for result in reused_results + fresh_results
        ), "OBJECT IDENTITY VIOLATION: Reused instance behaved differently than fresh instances"


class TestConcurrentExecutionSafety:
    """Test Requirement 5: Concurrent executions with same input yield same results."""

    def test_concurrent_execution_4_threads_20_calls(self) -> None:
        """Concurrent executions (4 threads, 20 calls) produce identical results."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "users": [{"name": "Alice", "id": 2}, {"name": "Bob", "id": 1}],
                "metadata": {"version": "1.0", "created": "2024-01-01"},
            }
        }
        context = {"correlation_id": "test-concurrent"}

        # Act: Execute concurrently with 4 threads, 20 total executions
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(plugin.execute, input_data, context) for _ in range(20)
            ]
            concurrent_results = [future.result() for future in futures]

        # Assert: All concurrent results are identical
        first_result = concurrent_results[0]
        assert all(
            result == first_result for result in concurrent_results
        ), "CONCURRENT EXECUTION VIOLATION: Parallel executions produced different results"

    def test_concurrent_execution_10_threads_50_calls(self) -> None:
        """High concurrency (10 threads, 50 calls) maintains determinism."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {f"key_{i}": i for i in range(20)}}
        context = {"correlation_id": "test-concurrent-high"}

        # Act: High concurrency execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(plugin.execute, input_data, context) for _ in range(50)
            ]
            concurrent_results = [future.result() for future in futures]

        # Assert: All results identical despite high concurrency
        first_result = concurrent_results[0]
        assert all(
            result == first_result for result in concurrent_results
        ), "CONCURRENT EXECUTION VIOLATION: High concurrency produced non-deterministic results"

    def test_concurrent_mixed_inputs(self) -> None:
        """Concurrent execution with different inputs produces consistent results."""
        # Arrange
        plugin = PluginJsonNormalizer()
        inputs = [
            {"json": {"a": 1}},
            {"json": {"b": 2}},
            {"json": {"c": 3}},
        ]
        context = {"correlation_id": "test-concurrent-mixed"}

        # Act: Execute each input concurrently multiple times
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Execute each input 10 times concurrently
            futures_input1 = [
                executor.submit(plugin.execute, inputs[0], context) for _ in range(10)
            ]
            futures_input2 = [
                executor.submit(plugin.execute, inputs[1], context) for _ in range(10)
            ]
            futures_input3 = [
                executor.submit(plugin.execute, inputs[2], context) for _ in range(10)
            ]

            results_input1 = [f.result() for f in futures_input1]
            results_input2 = [f.result() for f in futures_input2]
            results_input3 = [f.result() for f in futures_input3]

        # Assert: Each input group has identical results
        assert all(
            r == results_input1[0] for r in results_input1
        ), "CONCURRENT EXECUTION VIOLATION: Input 1 produced different results"
        assert all(
            r == results_input2[0] for r in results_input2
        ), "CONCURRENT EXECUTION VIOLATION: Input 2 produced different results"
        assert all(
            r == results_input3[0] for r in results_input3
        ), "CONCURRENT EXECUTION VIOLATION: Input 3 produced different results"


class TestCallOrderIndependence:
    """Test that plugin produces same results regardless of call order."""

    def test_determinism_across_different_call_orders(self) -> None:
        """Plugin produces same results regardless of call order."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input1 = {"json": {"a": 1, "b": 2}}
        input2 = {"json": {"x": 10, "y": 20}}
        input3 = {"json": {"p": 100, "q": 200}}
        context = {"correlation_id": "test-order"}

        # Act: Execute in different orders
        # Order 1: 1 -> 2 -> 3
        result1_order1 = plugin.execute(input1, context)
        result2_order1 = plugin.execute(input2, context)
        result3_order1 = plugin.execute(input3, context)

        # Order 2: 3 -> 1 -> 2
        result3_order2 = plugin.execute(input3, context)
        result1_order2 = plugin.execute(input1, context)
        result2_order2 = plugin.execute(input2, context)

        # Order 3: 2 -> 3 -> 1
        result2_order3 = plugin.execute(input2, context)
        result3_order3 = plugin.execute(input3, context)
        result1_order3 = plugin.execute(input1, context)

        # Assert: Same inputs produce same outputs regardless of call order
        assert (
            result1_order1 == result1_order2 == result1_order3
        ), "CALL ORDER VIOLATION: Input 1 results changed based on call order"
        assert (
            result2_order1 == result2_order2 == result2_order3
        ), "CALL ORDER VIOLATION: Input 2 results changed based on call order"
        assert (
            result3_order1 == result3_order2 == result3_order3
        ), "CALL ORDER VIOLATION: Input 3 results changed based on call order"

    def test_interleaved_calls_determinism(self) -> None:
        """Interleaved calls with different inputs maintain determinism."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_a = {"json": {"key": "a"}}
        input_b = {"json": {"key": "b"}}
        context = {"correlation_id": "test-interleave"}

        # Act: Interleaved execution pattern
        # Pattern 1: A, B, A, B, A, B
        results_a_pattern1 = []
        results_b_pattern1 = []
        for _ in range(3):
            results_a_pattern1.append(plugin.execute(input_a, context))
            results_b_pattern1.append(plugin.execute(input_b, context))

        # Pattern 2: B, A, B, A, B, A
        results_a_pattern2 = []
        results_b_pattern2 = []
        for _ in range(3):
            results_b_pattern2.append(plugin.execute(input_b, context))
            results_a_pattern2.append(plugin.execute(input_a, context))

        # Assert: Results identical regardless of interleaving pattern
        assert all(
            r == results_a_pattern1[0] for r in results_a_pattern1 + results_a_pattern2
        ), "CALL ORDER VIOLATION: Input A results varied with interleaving"
        assert all(
            r == results_b_pattern1[0] for r in results_b_pattern1 + results_b_pattern2
        ), "CALL ORDER VIOLATION: Input B results varied with interleaving"


class TestInputImmutability:
    """Test that plugin does not modify input_data or context."""

    def test_input_data_not_modified(self) -> None:
        """Plugin does not modify input_data dictionary."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"z": 3, "a": 1, "m": 2}}
        input_data_original = copy.deepcopy(input_data)
        context = {"correlation_id": "test-immutable"}

        # Act
        plugin.execute(input_data, context)

        # Assert: input_data unchanged
        assert (
            input_data == input_data_original
        ), "INPUT IMMUTABILITY VIOLATION: input_data was modified"

    def test_context_not_modified(self) -> None:
        """Plugin does not modify context dictionary."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"key": "value"}}
        context = {"correlation_id": "test-context", "metadata": {"test": "data"}}
        context_original = copy.deepcopy(context)

        # Act
        plugin.execute(input_data, context)

        # Assert: context unchanged
        assert (
            context == context_original
        ), "INPUT IMMUTABILITY VIOLATION: context was modified"

    def test_nested_input_not_modified(self) -> None:
        """Plugin does not modify nested structures in input_data."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "level1": {"level2": {"level3": {"value": [1, 2, 3]}}},
            }
        }
        input_data_original = copy.deepcopy(input_data)
        context = {"correlation_id": "test-nested-immutable"}

        # Act
        plugin.execute(input_data, context)

        # Assert: Nested structures unchanged
        assert (
            input_data == input_data_original
        ), "INPUT IMMUTABILITY VIOLATION: Nested input structures were modified"


class TestStateIndependence:
    """Test that plugin does not maintain mutable state between calls."""

    def test_no_state_mutation_between_calls(self) -> None:
        """Plugin does not maintain mutable state between calls."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"key": "value"}}
        context = {"correlation_id": "test-state"}

        # Act: Execute multiple times with same input
        result1 = plugin.execute(input_data, context)
        result2 = plugin.execute(input_data, context)
        result3 = plugin.execute(input_data, context)

        # Assert: All results identical (no state accumulation)
        assert (
            result1 == result2 == result3
        ), "STATE INDEPENDENCE VIOLATION: Results changed across calls (state accumulation detected)"

    def test_alternating_inputs_no_state_leakage(self) -> None:
        """Alternating different inputs shows no state leakage."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input1 = {"json": {"type": "first"}}
        input2 = {"json": {"type": "second"}}
        context = {"correlation_id": "test-leakage"}

        # Act: Alternate between inputs
        result1_first = plugin.execute(input1, context)
        result2_first = plugin.execute(input2, context)
        result1_second = plugin.execute(input1, context)
        result2_second = plugin.execute(input2, context)
        result1_third = plugin.execute(input1, context)

        # Assert: Results for each input are identical (no state leakage)
        assert (
            result1_first == result1_second == result1_third
        ), "STATE LEAKAGE VIOLATION: Input 1 results changed"
        assert (
            result2_first == result2_second
        ), "STATE LEAKAGE VIOLATION: Input 2 results changed"


class TestDeepCopyIndependence:
    """Test that plugin works correctly with deep-copied inputs."""

    def test_deep_copy_independence(self) -> None:
        """Plugin works correctly with deep-copied inputs."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"nested": {"deep": {"value": 42}}}}
        context = {"correlation_id": "test-deepcopy"}

        # Act: Execute with original and deep-copied inputs
        result_original = plugin.execute(input_data, context)
        result_deepcopy = plugin.execute(copy.deepcopy(input_data), context)

        # Assert: Results identical regardless of copy status
        assert (
            result_original == result_deepcopy
        ), "DEEP COPY VIOLATION: Deep-copied inputs produced different results"

    def test_multiple_deep_copies_identical_results(self) -> None:
        """Multiple deep copies produce identical results."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {"json": {"complex": {"structure": [1, 2, 3]}}}
        context = {"correlation_id": "test-multi-deepcopy"}

        # Act: Execute with multiple deep copies
        results = [
            plugin.execute(copy.deepcopy(input_data), context) for _ in range(10)
        ]

        # Assert: All deep copy results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "DEEP COPY VIOLATION: Multiple deep copies produced different results"


class TestEdgeCases:
    """Test determinism with edge case inputs."""

    def test_empty_json_determinism(self) -> None:
        """Empty JSON input produces deterministic results."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data: dict[str, Any] = {"json": {}}
        context = {"correlation_id": "test-empty"}

        # Act: Execute multiple times
        results = [plugin.execute(input_data, context) for _ in range(50)]

        # Assert: All results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "EDGE CASE VIOLATION: Empty input produced non-deterministic results"

    def test_large_nested_structure_determinism(self) -> None:
        """Large nested structure produces deterministic results."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                f"level1_{i}": {
                    f"level2_{j}": {f"level3_{k}": k for k in range(5)}
                    for j in range(5)
                }
                for i in range(5)
            }
        }
        context = {"correlation_id": "test-large-nested"}

        # Act: Execute multiple times
        results = [plugin.execute(input_data, context) for _ in range(20)]

        # Assert: All results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "EDGE CASE VIOLATION: Large nested structure produced non-deterministic results"

    def test_unicode_input_determinism(self) -> None:
        """Unicode input produces deterministic results."""
        # Arrange
        plugin = PluginJsonNormalizer()
        input_data = {
            "json": {
                "emoji": "🔥🎉🚀",
                "chinese": "你好世界",
                "arabic": "مرحبا بالعالم",
                "mixed": "Hello 世界 🌍",
            }
        }
        context = {"correlation_id": "test-unicode"}

        # Act: Execute multiple times
        results = [plugin.execute(input_data, context) for _ in range(50)]

        # Assert: All results identical
        first_result = results[0]
        assert all(
            result == first_result for result in results
        ), "EDGE CASE VIOLATION: Unicode input produced non-deterministic results"

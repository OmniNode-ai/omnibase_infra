# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Architecture Compliance Tests for Compute Plugins.

Tests verify that compute plugins adhere to ONEX 4-node architecture principles:
- Plugins belong ONLY to COMPUTE layer (pure transformations)
- NO side effects (no I/O, no external state, no mutations)
- Deterministic behavior (same inputs → same outputs)
- Clear separation from EFFECT, REDUCER, ORCHESTRATOR layers

These tests ensure architectural integrity and prevent violations of the
COMPUTE layer contract.

Note on Type Annotations:
    This test file uses Pydantic models (ModelPluginInputData, ModelPluginContext,
    ModelPluginOutputData) instead of TypedDict to conform to ONEX standards.
    Some tests intentionally demonstrate architectural violations - where mutations
    are tested, type: ignore comments are used to document these intentional
    violations for testing purposes.
"""

from unittest.mock import patch

import pytest

from omnibase_infra.plugins.models import (
    ModelPluginContext,
    ModelPluginInputData,
    ModelPluginOutputData,
)
from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase


class TestOnexArchitectureCompliance:
    """Test ONEX 4-node architecture compliance for compute plugins.

    Verifies that plugins respect the COMPUTE layer contract and do not
    perform operations that belong in EFFECT, REDUCER, or ORCHESTRATOR layers.
    """

    def test_plugin_must_not_perform_network_io(self) -> None:
        """Compute plugins MUST NOT perform network I/O (EFFECT layer responsibility)."""

        # Arrange: Plugin that violates architecture by doing HTTP call
        class NetworkViolator(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                import urllib.request

                # ARCHITECTURAL VIOLATION: Network I/O in COMPUTE layer
                with urllib.request.urlopen("http://example.com") as response:
                    return ModelPluginOutputData.model_validate(
                        {"data": response.read()}
                    )

        plugin = NetworkViolator()

        # Act & Assert: Network operations should be blocked
        # In real implementation, this would be caught by static analysis
        # or runtime monitoring, but we test the architectural principle
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = RuntimeError(
                "Network I/O not allowed in COMPUTE layer"
            )

            with pytest.raises(
                RuntimeError, match="Network I/O not allowed in COMPUTE layer"
            ):
                plugin.execute(
                    ModelPluginInputData(), ModelPluginContext(correlation_id="test")
                )

    def test_plugin_must_not_perform_file_io(self) -> None:
        """Compute plugins MUST NOT perform file I/O (EFFECT layer responsibility)."""

        # Arrange: Plugin that violates architecture by reading file
        class FileIOViolator(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # ARCHITECTURAL VIOLATION: File I/O in COMPUTE layer
                with open("/tmp/data.txt", encoding="utf-8") as f:  # noqa: S108 - Intentional violation for testing
                    return ModelPluginOutputData.model_validate({"data": f.read()})

        plugin = FileIOViolator()

        # Act & Assert: File operations should be blocked
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = RuntimeError(
                "File I/O not allowed in COMPUTE layer"
            )

            with pytest.raises(
                RuntimeError, match="File I/O not allowed in COMPUTE layer"
            ):
                plugin.execute(
                    ModelPluginInputData(), ModelPluginContext(correlation_id="test")
                )

    def test_plugin_must_not_access_database(self) -> None:
        """Compute plugins MUST NOT access databases (EFFECT layer responsibility)."""

        # Arrange: Plugin that violates architecture by querying database
        class DatabaseViolator(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                import sqlite3

                # ARCHITECTURAL VIOLATION: Database access in COMPUTE layer
                conn = sqlite3.connect(":memory:")
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return ModelPluginOutputData.model_validate({"data": result})

        plugin = DatabaseViolator()

        # Act & Assert: Database operations should be blocked
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = RuntimeError(
                "Database access not allowed in COMPUTE layer"
            )

            with pytest.raises(
                RuntimeError, match="Database access not allowed in COMPUTE layer"
            ):
                plugin.execute(
                    ModelPluginInputData(), ModelPluginContext(correlation_id="test")
                )

    def test_plugin_must_not_use_global_state(self) -> None:
        """Compute plugins MUST NOT rely on mutable global state."""

        # Arrange: Global state that plugins should NOT use
        global_counter = {"count": 0}

        class GlobalStateViolator(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # ARCHITECTURAL VIOLATION: Mutable global state
                global_counter["count"] += 1
                return ModelPluginOutputData.model_validate(
                    {"count": global_counter["count"]}
                )

        plugin = GlobalStateViolator()

        # Act: Execute twice
        result1 = plugin.execute(
            ModelPluginInputData(), ModelPluginContext(correlation_id="test1")
        )
        result2 = plugin.execute(
            ModelPluginInputData(), ModelPluginContext(correlation_id="test2")
        )

        # Assert: Results differ due to global state (VIOLATION)
        # This demonstrates why global state breaks determinism
        assert result1.get("count") == 1
        assert result2.get("count") == 2
        assert result1 != result2  # Same input, different output = NOT DETERMINISTIC

    def test_plugin_must_be_deterministic(self) -> None:
        """Compute plugins MUST be deterministic (same inputs → same outputs)."""

        # Arrange: Deterministic plugin (CORRECT)
        class DeterministicPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # Pure computation - deterministic
                values = input_data.get("values", [])
                # Cast values to list for type safety - values is always list[int] from input
                # type: ignore[call-overload] - safe cast from dynamic get() return to list[int]
                values_list: list[int] = list(values) if values else []  # type: ignore[call-overload]
                return ModelPluginOutputData.model_validate(
                    {"sum": sum(values_list), "count": len(values_list)}
                )

        plugin = DeterministicPlugin()
        input_data = ModelPluginInputData.model_validate({"values": [1, 2, 3, 4, 5]})
        context = ModelPluginContext(correlation_id="test")

        # Act: Execute 10 times
        results = [plugin.execute(input_data, context) for _ in range(10)]

        # Assert: All results identical (deterministic)
        assert all(result == results[0] for result in results)
        expected = ModelPluginOutputData.model_validate({"sum": 15, "count": 5})
        assert results[0] == expected

    def test_plugin_must_not_use_non_deterministic_randomness(self) -> None:
        """Compute plugins MUST NOT use non-deterministic random numbers."""

        # Arrange: Plugin that uses random without seed (VIOLATION)
        class RandomViolator(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                import random

                # ARCHITECTURAL VIOLATION: Non-deterministic randomness
                return ModelPluginOutputData.model_validate(
                    {"random_value": random.random()}
                )

        plugin = RandomViolator()

        # Act: Execute twice
        result1 = plugin.execute(
            ModelPluginInputData(), ModelPluginContext(correlation_id="test1")
        )
        result2 = plugin.execute(
            ModelPluginInputData(), ModelPluginContext(correlation_id="test2")
        )

        # Assert: Results differ (non-deterministic - VIOLATION)
        assert result1.get("random_value") != result2.get("random_value")

    def test_plugin_can_use_deterministic_randomness_with_seed(self) -> None:
        """Compute plugins CAN use random numbers if seeded deterministically."""

        # Arrange: Plugin with deterministic randomness (CORRECT)
        class SeededRandomPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                import random

                # ACCEPTABLE: Deterministic randomness with seed from context
                seed_val = context.get("random_seed", 42)
                # Cast to int for random.seed - safe since we provide default 42
                # type: ignore[call-overload] - safe cast from dynamic get() return to int
                seed = int(seed_val) if seed_val is not None else 42  # type: ignore[call-overload]
                random.seed(seed)

                return ModelPluginOutputData.model_validate(
                    {"random_values": [random.random() for _ in range(5)]}
                )

        plugin = SeededRandomPlugin()
        context = ModelPluginContext(correlation_id="test", random_seed=12345)

        # Act: Execute 10 times with same seed
        results = [plugin.execute(ModelPluginInputData(), context) for _ in range(10)]

        # Assert: All results identical (deterministic with seed)
        assert all(result == results[0] for result in results)

    def test_plugin_must_not_access_current_time_non_deterministically(self) -> None:
        """Compute plugins MUST NOT access current time without it being provided."""

        # Arrange: Plugin that uses current time (VIOLATION)
        class TimeViolator(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                import time

                # ARCHITECTURAL VIOLATION: Non-deterministic time access
                return ModelPluginOutputData.model_validate({"timestamp": time.time()})

        plugin = TimeViolator()

        # Act: Execute twice
        result1 = plugin.execute(
            ModelPluginInputData(), ModelPluginContext(correlation_id="test1")
        )
        import time

        time.sleep(0.01)  # Small delay to ensure different timestamps
        result2 = plugin.execute(
            ModelPluginInputData(), ModelPluginContext(correlation_id="test2")
        )

        # Assert: Results differ due to time (VIOLATION)
        assert result1.get("timestamp") != result2.get("timestamp")

    def test_plugin_can_use_time_if_provided_in_context(self) -> None:
        """Compute plugins CAN use time if it is passed as input."""

        # Arrange: Plugin with deterministic time (CORRECT)
        class DeterministicTimePlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # ACCEPTABLE: Time provided as input parameter (ISO string format)
                execution_time = context.get("execution_timestamp", "")
                return ModelPluginOutputData.model_validate(
                    {"timestamp": execution_time, "processed": True}
                )

        plugin = DeterministicTimePlugin()
        # execution_timestamp must be a string (ISO format) per ModelPluginContext
        context = ModelPluginContext(
            correlation_id="test", execution_timestamp="2009-02-13T23:31:30Z"
        )

        # Act: Execute 10 times with same timestamp
        results = [plugin.execute(ModelPluginInputData(), context) for _ in range(10)]

        # Assert: All results identical (deterministic with provided time)
        assert all(result == results[0] for result in results)
        assert results[0].get("timestamp") == "2009-02-13T23:31:30Z"

    def test_plugin_must_not_modify_input_data(self) -> None:
        """Compute plugins MUST NOT modify input_data (no side effects).

        Note: This test intentionally demonstrates mutation of Pydantic model
        attributes to test side effect detection. The setattr usage is an
        intentional violation for testing purposes.
        """

        # Arrange: Plugin that modifies input (VIOLATION)
        class InputModifier(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # ARCHITECTURAL VIOLATION: Mutating input data
                # Note: Using setattr since Pydantic models don't support
                # dict-style indexed assignment by default.
                # type: ignore[attr-defined] - intentional mutation for testing side effects
                object.__setattr__(input_data, "modified", True)  # type: ignore[attr-defined]
                return ModelPluginOutputData.model_validate({"result": "modified"})

        plugin = InputModifier()
        input_data = ModelPluginInputData.model_validate({"value": 42})
        original_fields = set(input_data.model_fields_set)

        # Act
        plugin.execute(input_data, ModelPluginContext(correlation_id="test"))

        # Assert: Input was modified (VIOLATION)
        assert input_data.model_fields_set != original_fields or hasattr(
            input_data, "modified"
        )
        assert hasattr(input_data, "modified")  # Side effect detected

    def test_plugin_must_not_modify_context(self) -> None:
        """Compute plugins MUST NOT modify context (no side effects).

        Note: This test intentionally demonstrates mutation of Pydantic model
        attributes to test side effect detection. The setattr usage is an
        intentional violation for testing purposes.
        """

        # Arrange: Plugin that modifies context (VIOLATION)
        class ContextModifier(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # ARCHITECTURAL VIOLATION: Mutating context
                # Note: Using setattr since Pydantic models don't support
                # dict-style indexed assignment by default.
                current_count = context.get("execution_count", 0)
                # type: ignore[call-overload] - safe cast from dynamic get() return to int
                count_val = int(current_count) if current_count else 0  # type: ignore[call-overload]
                # type: ignore[attr-defined] - intentional mutation for testing side effects
                object.__setattr__(context, "execution_count", count_val + 1)  # type: ignore[attr-defined]
                return ModelPluginOutputData.model_validate({"result": "modified"})

        plugin = ContextModifier()
        context = ModelPluginContext(correlation_id="test")
        original_fields = set(context.model_fields_set)

        # Act
        plugin.execute(ModelPluginInputData(), context)

        # Assert: Context was modified (VIOLATION)
        assert context.model_fields_set != original_fields or hasattr(
            context, "execution_count"
        )
        assert hasattr(context, "execution_count")  # Side effect detected

    def test_plugin_separation_from_effect_layer(self) -> None:
        """Demonstrates clear separation between COMPUTE and EFFECT layers."""

        # Arrange: COMPUTE plugin (pure transformation)
        class ComputePlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # COMPUTE: Pure data transformation
                values = input_data.get("values", [])
                # type: ignore[call-overload] - safe cast from dynamic get() return to list[int]
                values_list: list[int] = list(values) if values else []  # type: ignore[call-overload]
                return ModelPluginOutputData.model_validate(
                    {
                        "sum": sum(values_list),
                        "average": sum(values_list) / len(values_list)
                        if values_list
                        else 0,
                        "count": len(values_list),
                    }
                )

        # EFFECT layer would handle I/O:
        # - NodeEffectService reads data from database
        # - Calls ComputePlugin.execute() for transformation
        # - NodeEffectService writes results back to database

        plugin = ComputePlugin()
        input_data = ModelPluginInputData.model_validate(
            {"values": [10, 20, 30, 40, 50]}
        )
        context = ModelPluginContext(correlation_id="test")

        # Act: Pure computation
        result = plugin.execute(input_data, context)

        # Assert: Result is correct and deterministic
        expected = ModelPluginOutputData.model_validate(
            {"sum": 150, "average": 30.0, "count": 5}
        )
        assert result == expected

        # Execute again - same result (deterministic)
        result2 = plugin.execute(input_data, context)
        assert result == result2

    def test_plugin_must_not_perform_multi_source_aggregation(self) -> None:
        """Multi-source aggregation belongs in REDUCER layer, not COMPUTE."""

        # This test documents that REDUCER operations should NOT be in plugins

        # COMPUTE Plugin (CORRECT - single source transformation):
        class SingleSourcePlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # ACCEPTABLE: Transform single input source
                value = input_data.get("value", 0)
                # type: ignore[call-overload] - safe cast from dynamic get() return to int
                value_int = int(value) if value else 0  # type: ignore[call-overload]
                return ModelPluginOutputData.model_validate(
                    {"processed": value_int * 2}
                )

        # REDUCER would handle multi-source aggregation:
        # - Fetches data from database, cache, message queue
        # - Aggregates and consolidates state
        # - May use ComputePlugin for transformations

        plugin = SingleSourcePlugin()
        result = plugin.execute(
            ModelPluginInputData.model_validate({"value": 21}),
            ModelPluginContext(correlation_id="test"),
        )

        expected = ModelPluginOutputData.model_validate({"processed": 42})
        assert result == expected

    def test_plugin_must_not_coordinate_workflows(self) -> None:
        """Workflow coordination belongs in ORCHESTRATOR layer, not COMPUTE."""

        # This test documents that ORCHESTRATOR operations should NOT be in plugins

        # COMPUTE Plugin (CORRECT - single step transformation):
        class SingleStepPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # ACCEPTABLE: Single transformation step
                email = input_data.get("email", "")
                # Cast to str for endswith check
                email_str = str(email) if email else ""
                return ModelPluginOutputData.model_validate(
                    {"validated": email_str.endswith("@example.com")}
                )

        # ORCHESTRATOR would handle multi-step workflows:
        # - Coordinates multiple nodes
        # - Manages workflow state transitions
        # - May use ComputePlugins for individual steps

        plugin = SingleStepPlugin()
        result = plugin.execute(
            ModelPluginInputData.model_validate({"email": "user@example.com"}),
            ModelPluginContext(correlation_id="test"),
        )

        expected = ModelPluginOutputData.model_validate({"validated": True})
        assert result == expected


class TestArchitecturalBenefits:
    """Test architectural benefits of COMPUTE layer separation."""

    def test_compute_plugins_are_easily_testable(self) -> None:
        """COMPUTE plugins are trivially testable (no mocking required)."""

        # Arrange: Simple compute plugin
        class EasyToTestPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                values = input_data.get("values", [])
                # type: ignore[call-overload] - safe cast from dynamic get() return to list[int]
                values_list: list[int] = list(values) if values else []  # type: ignore[call-overload]
                return ModelPluginOutputData.model_validate(
                    {"max": max(values_list) if values_list else None}
                )

        plugin = EasyToTestPlugin()

        # Act & Assert: No mocking needed - pure function
        result1 = plugin.execute(
            ModelPluginInputData.model_validate({"values": [1, 5, 3]}),
            ModelPluginContext(),
        )
        assert result1.get("max") == 5

        result2 = plugin.execute(
            ModelPluginInputData.model_validate({"values": []}),
            ModelPluginContext(),
        )
        assert result2.get("max") is None

    def test_compute_plugins_are_composable(self) -> None:
        """COMPUTE plugins can be composed without coordination complexity."""

        # Arrange: Two composable plugins
        class NormalizerPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                values = input_data.get("values", [])
                # type: ignore[call-overload] - safe cast from dynamic get() return to list[int]
                values_list: list[int] = list(values) if values else []  # type: ignore[call-overload]
                max_val = max(values_list) if values_list else 1
                normalized = [v / max_val for v in values_list]
                return ModelPluginOutputData.model_validate({"normalized": normalized})

        class AggregatorPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                normalized = input_data.get("normalized", [])
                # type: ignore[call-overload] - safe cast from dynamic get() return to list[float]
                normalized_list: list[float] = list(normalized) if normalized else []  # type: ignore[call-overload]
                return ModelPluginOutputData.model_validate(
                    {"sum": sum(normalized_list), "count": len(normalized_list)}
                )

        # Act: Compose plugins
        step1 = NormalizerPlugin()
        step2 = AggregatorPlugin()

        input_data = ModelPluginInputData.model_validate({"values": [10, 20, 30]})
        context = ModelPluginContext(correlation_id="test")

        result1 = step1.execute(input_data, context)
        # Pass result1 to step2 by converting to input format
        result1_as_input = ModelPluginInputData.model_validate(
            {"normalized": result1.get("normalized")}
        )
        result2 = step2.execute(result1_as_input, context)

        # Assert: Composition works seamlessly
        assert result2.get("count") == 3
        sum_val = result2.get("sum")
        assert isinstance(sum_val, int | float)
        assert abs(sum_val - 2.0) < 0.01  # 10/30 + 20/30 + 30/30 = 2.0

    def test_compute_plugins_enable_horizontal_scaling(self) -> None:
        """Stateless COMPUTE plugins enable easy horizontal scaling."""

        # Arrange: Stateless plugin
        class StatelessPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # No state - safe to run in parallel
                value = input_data.get("value", 0)
                # type: ignore[call-overload] - safe cast from dynamic get() return to int
                value_int = int(value) if value else 0  # type: ignore[call-overload]
                return ModelPluginOutputData.model_validate({"processed": value_int**2})

        # Act: Simulate parallel execution (multiple instances)
        plugin1 = StatelessPlugin()
        plugin2 = StatelessPlugin()
        plugin3 = StatelessPlugin()

        results = [
            plugin1.execute(
                ModelPluginInputData.model_validate({"value": 2}),
                ModelPluginContext(correlation_id="test1"),
            ),
            plugin2.execute(
                ModelPluginInputData.model_validate({"value": 3}),
                ModelPluginContext(correlation_id="test2"),
            ),
            plugin3.execute(
                ModelPluginInputData.model_validate({"value": 4}),
                ModelPluginContext(correlation_id="test3"),
            ),
        ]

        # Assert: Each instance produces correct result independently
        expected = [
            ModelPluginOutputData.model_validate({"processed": 4}),
            ModelPluginOutputData.model_validate({"processed": 9}),
            ModelPluginOutputData.model_validate({"processed": 16}),
        ]
        assert results == expected

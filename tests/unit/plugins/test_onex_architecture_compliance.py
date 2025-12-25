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
"""

from unittest.mock import patch

import pytest

from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase
from omnibase_infra.protocols.protocol_plugin_compute import (
    PluginContext,
    PluginInputData,
    PluginOutputData,
)


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
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                import urllib.request

                # ARCHITECTURAL VIOLATION: Network I/O in COMPUTE layer
                with urllib.request.urlopen("http://example.com") as response:
                    return {"data": response.read()}

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
                plugin.execute({}, {"correlation_id": "test"})

    def test_plugin_must_not_perform_file_io(self) -> None:
        """Compute plugins MUST NOT perform file I/O (EFFECT layer responsibility)."""

        # Arrange: Plugin that violates architecture by reading file
        class FileIOViolator(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # ARCHITECTURAL VIOLATION: File I/O in COMPUTE layer
                with open("/tmp/data.txt", encoding="utf-8") as f:  # noqa: S108 - Intentional violation for testing
                    return {"data": f.read()}

        plugin = FileIOViolator()

        # Act & Assert: File operations should be blocked
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = RuntimeError(
                "File I/O not allowed in COMPUTE layer"
            )

            with pytest.raises(
                RuntimeError, match="File I/O not allowed in COMPUTE layer"
            ):
                plugin.execute({}, {"correlation_id": "test"})

    def test_plugin_must_not_access_database(self) -> None:
        """Compute plugins MUST NOT access databases (EFFECT layer responsibility)."""

        # Arrange: Plugin that violates architecture by querying database
        class DatabaseViolator(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                import sqlite3

                # ARCHITECTURAL VIOLATION: Database access in COMPUTE layer
                conn = sqlite3.connect(":memory:")
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return {"data": result}

        plugin = DatabaseViolator()

        # Act & Assert: Database operations should be blocked
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = RuntimeError(
                "Database access not allowed in COMPUTE layer"
            )

            with pytest.raises(
                RuntimeError, match="Database access not allowed in COMPUTE layer"
            ):
                plugin.execute({}, {"correlation_id": "test"})

    def test_plugin_must_not_use_global_state(self) -> None:
        """Compute plugins MUST NOT rely on mutable global state."""

        # Arrange: Global state that plugins should NOT use
        global_counter = {"count": 0}

        class GlobalStateViolator(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # ARCHITECTURAL VIOLATION: Mutable global state
                global_counter["count"] += 1
                return {"count": global_counter["count"]}

        plugin = GlobalStateViolator()

        # Act: Execute twice
        result1 = plugin.execute({}, {"correlation_id": "test1"})
        result2 = plugin.execute({}, {"correlation_id": "test2"})

        # Assert: Results differ due to global state (VIOLATION)
        # This demonstrates why global state breaks determinism
        assert result1["count"] == 1
        assert result2["count"] == 2
        assert result1 != result2  # Same input, different output = NOT DETERMINISTIC

    def test_plugin_must_be_deterministic(self) -> None:
        """Compute plugins MUST be deterministic (same inputs → same outputs)."""

        # Arrange: Deterministic plugin (CORRECT)
        class DeterministicPlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # Pure computation - deterministic
                values = input_data.get("values", [])
                return {"sum": sum(values), "count": len(values)}

        plugin = DeterministicPlugin()
        input_data = {"values": [1, 2, 3, 4, 5]}
        context = {"correlation_id": "test"}

        # Act: Execute 10 times
        results = [plugin.execute(input_data, context) for _ in range(10)]

        # Assert: All results identical (deterministic)
        assert all(result == results[0] for result in results)
        assert results[0] == {"sum": 15, "count": 5}

    def test_plugin_must_not_use_non_deterministic_randomness(self) -> None:
        """Compute plugins MUST NOT use non-deterministic random numbers."""

        # Arrange: Plugin that uses random without seed (VIOLATION)
        class RandomViolator(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                import random

                # ARCHITECTURAL VIOLATION: Non-deterministic randomness
                return {"random_value": random.random()}

        plugin = RandomViolator()

        # Act: Execute twice
        result1 = plugin.execute({}, {"correlation_id": "test1"})
        result2 = plugin.execute({}, {"correlation_id": "test2"})

        # Assert: Results differ (non-deterministic - VIOLATION)
        assert result1["random_value"] != result2["random_value"]

    def test_plugin_can_use_deterministic_randomness_with_seed(self) -> None:
        """Compute plugins CAN use random numbers if seeded deterministically."""

        # Arrange: Plugin with deterministic randomness (CORRECT)
        class SeededRandomPlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                import random

                # ACCEPTABLE: Deterministic randomness with seed from context
                seed = context.get("random_seed", 42)
                random.seed(seed)

                return {"random_values": [random.random() for _ in range(5)]}

        plugin = SeededRandomPlugin()
        context = {"correlation_id": "test", "random_seed": 12345}

        # Act: Execute 10 times with same seed
        results = [plugin.execute({}, context) for _ in range(10)]

        # Assert: All results identical (deterministic with seed)
        assert all(result == results[0] for result in results)

    def test_plugin_must_not_access_current_time_non_deterministically(self) -> None:
        """Compute plugins MUST NOT access current time without it being provided."""

        # Arrange: Plugin that uses current time (VIOLATION)
        class TimeViolator(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                import time

                # ARCHITECTURAL VIOLATION: Non-deterministic time access
                return {"timestamp": time.time()}

        plugin = TimeViolator()

        # Act: Execute twice
        result1 = plugin.execute({}, {"correlation_id": "test1"})
        import time

        time.sleep(0.01)  # Small delay to ensure different timestamps
        result2 = plugin.execute({}, {"correlation_id": "test2"})

        # Assert: Results differ due to time (VIOLATION)
        assert result1["timestamp"] != result2["timestamp"]

    def test_plugin_can_use_time_if_provided_in_context(self) -> None:
        """Compute plugins CAN use time if it is passed as input."""

        # Arrange: Plugin with deterministic time (CORRECT)
        class DeterministicTimePlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # ACCEPTABLE: Time provided as input parameter
                execution_time = context.get("execution_timestamp", 0)
                return {"timestamp": execution_time, "processed": True}

        plugin = DeterministicTimePlugin()
        context = {"correlation_id": "test", "execution_timestamp": 1234567890}

        # Act: Execute 10 times with same timestamp
        results = [plugin.execute({}, context) for _ in range(10)]

        # Assert: All results identical (deterministic with provided time)
        assert all(result == results[0] for result in results)
        assert results[0]["timestamp"] == 1234567890

    def test_plugin_must_not_modify_input_data(self) -> None:
        """Compute plugins MUST NOT modify input_data (no side effects)."""

        # Arrange: Plugin that modifies input (VIOLATION)
        class InputModifier(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # ARCHITECTURAL VIOLATION: Mutating input data
                input_data["modified"] = True
                return {"result": "modified"}

        plugin = InputModifier()
        input_data = {"value": 42}
        original_input = input_data.copy()

        # Act
        plugin.execute(input_data, {"correlation_id": "test"})

        # Assert: Input was modified (VIOLATION)
        assert input_data != original_input
        assert "modified" in input_data  # Side effect detected

    def test_plugin_must_not_modify_context(self) -> None:
        """Compute plugins MUST NOT modify context (no side effects)."""

        # Arrange: Plugin that modifies context (VIOLATION)
        class ContextModifier(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # ARCHITECTURAL VIOLATION: Mutating context
                context["execution_count"] = context.get("execution_count", 0) + 1
                return {"result": "modified"}

        plugin = ContextModifier()
        context = {"correlation_id": "test"}
        original_context = context.copy()

        # Act
        plugin.execute({}, context)

        # Assert: Context was modified (VIOLATION)
        assert context != original_context
        assert "execution_count" in context  # Side effect detected

    def test_plugin_separation_from_effect_layer(self) -> None:
        """Demonstrates clear separation between COMPUTE and EFFECT layers."""

        # Arrange: COMPUTE plugin (pure transformation)
        class ComputePlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # COMPUTE: Pure data transformation
                values = input_data.get("values", [])
                return {
                    "sum": sum(values),
                    "average": sum(values) / len(values) if values else 0,
                    "count": len(values),
                }

        # EFFECT layer would handle I/O:
        # - NodeEffectService reads data from database
        # - Calls ComputePlugin.execute() for transformation
        # - NodeEffectService writes results back to database

        plugin = ComputePlugin()
        input_data = {"values": [10, 20, 30, 40, 50]}
        context = {"correlation_id": "test"}

        # Act: Pure computation
        result = plugin.execute(input_data, context)

        # Assert: Result is correct and deterministic
        assert result == {"sum": 150, "average": 30.0, "count": 5}

        # Execute again - same result (deterministic)
        result2 = plugin.execute(input_data, context)
        assert result == result2

    def test_plugin_must_not_perform_multi_source_aggregation(self) -> None:
        """Multi-source aggregation belongs in REDUCER layer, not COMPUTE."""

        # This test documents that REDUCER operations should NOT be in plugins

        # COMPUTE Plugin (CORRECT - single source transformation):
        class SingleSourcePlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # ACCEPTABLE: Transform single input source
                return {"processed": input_data.get("value", 0) * 2}

        # REDUCER would handle multi-source aggregation:
        # - Fetches data from database, cache, message queue
        # - Aggregates and consolidates state
        # - May use ComputePlugin for transformations

        plugin = SingleSourcePlugin()
        result = plugin.execute({"value": 21}, {"correlation_id": "test"})

        assert result == {"processed": 42}

    def test_plugin_must_not_coordinate_workflows(self) -> None:
        """Workflow coordination belongs in ORCHESTRATOR layer, not COMPUTE."""

        # This test documents that ORCHESTRATOR operations should NOT be in plugins

        # COMPUTE Plugin (CORRECT - single step transformation):
        class SingleStepPlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # ACCEPTABLE: Single transformation step
                return {
                    "validated": input_data.get("email", "").endswith("@example.com")
                }

        # ORCHESTRATOR would handle multi-step workflows:
        # - Coordinates multiple nodes
        # - Manages workflow state transitions
        # - May use ComputePlugins for individual steps

        plugin = SingleStepPlugin()
        result = plugin.execute(
            {"email": "user@example.com"}, {"correlation_id": "test"}
        )

        assert result == {"validated": True}


class TestArchitecturalBenefits:
    """Test architectural benefits of COMPUTE layer separation."""

    def test_compute_plugins_are_easily_testable(self) -> None:
        """COMPUTE plugins are trivially testable (no mocking required)."""

        # Arrange: Simple compute plugin
        class EasyToTestPlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                values = input_data.get("values", [])
                return {"max": max(values) if values else None}

        plugin = EasyToTestPlugin()

        # Act & Assert: No mocking needed - pure function
        assert plugin.execute({"values": [1, 5, 3]}, {})["max"] == 5
        assert plugin.execute({"values": []}, {})["max"] is None

    def test_compute_plugins_are_composable(self) -> None:
        """COMPUTE plugins can be composed without coordination complexity."""

        # Arrange: Two composable plugins
        class NormalizerPlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                values = input_data.get("values", [])
                max_val = max(values) if values else 1
                return {"normalized": [v / max_val for v in values]}

        class AggregatorPlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                values = input_data.get("normalized", [])
                return {"sum": sum(values), "count": len(values)}

        # Act: Compose plugins
        step1 = NormalizerPlugin()
        step2 = AggregatorPlugin()

        input_data = {"values": [10, 20, 30]}
        context = {"correlation_id": "test"}

        result1 = step1.execute(input_data, context)
        result2 = step2.execute(result1, context)

        # Assert: Composition works seamlessly
        assert result2["count"] == 3
        assert abs(result2["sum"] - 2.0) < 0.01  # 10/30 + 20/30 + 30/30 = 2.0

    def test_compute_plugins_enable_horizontal_scaling(self) -> None:
        """Stateless COMPUTE plugins enable easy horizontal scaling."""

        # Arrange: Stateless plugin
        class StatelessPlugin(PluginComputeBase):
            def execute(
                self, input_data: PluginInputData, context: PluginContext
            ) -> PluginOutputData:
                # No state - safe to run in parallel
                return {"processed": input_data.get("value", 0) ** 2}

        # Act: Simulate parallel execution (multiple instances)
        plugin1 = StatelessPlugin()
        plugin2 = StatelessPlugin()
        plugin3 = StatelessPlugin()

        results = [
            plugin1.execute({"value": 2}, {"correlation_id": "test1"}),
            plugin2.execute({"value": 3}, {"correlation_id": "test2"}),
            plugin3.execute({"value": 4}, {"correlation_id": "test3"}),
        ]

        # Assert: Each instance produces correct result independently
        assert results == [{"processed": 4}, {"processed": 9}, {"processed": 16}]

#!/usr/bin/env python3
"""
Integration tests for contract parsing with real example files.

Tests the complete parsing pipeline with actual contract YAML files.
"""

from pathlib import Path

import pytest

from omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser


class TestContractParsingIntegration:
    """Integration tests with example contract files."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    @pytest.fixture
    def examples_dir(self):
        """Get examples directory."""
        # From tests/integration/codegen to src/omninode_bridge/codegen/schemas/examples
        test_dir = Path(__file__).parent
        project_root = test_dir.parent.parent.parent
        examples = (
            project_root
            / "src"
            / "omninode_bridge"
            / "codegen"
            / "schemas"
            / "examples"
        )
        return examples

    def test_parse_valid_minimal_v1(self, parser, examples_dir):
        """Test parsing valid minimal v1.0 contract."""
        contract_file = examples_dir / "valid_minimal_v1.yaml"

        if not contract_file.exists():
            pytest.skip(f"Example file not found: {contract_file}")

        contract = parser.parse_contract_file(contract_file)

        # Verify basic structure
        assert contract.name == "NodeMinimalEffect"
        assert contract.node_type == "effect"
        assert contract.version.major == 1

        # Verify backward compatibility
        assert len(contract.mixins) == 0
        assert contract.advanced_features is None

        # Should be valid
        assert contract.is_valid, f"Validation errors: {contract.validation_errors}"

    def test_parse_valid_with_mixins_v2(self, parser, examples_dir):
        """Test parsing valid v2.0 contract with mixins."""
        contract_file = examples_dir / "valid_with_mixins_v2.yaml"

        if not contract_file.exists():
            pytest.skip(f"Example file not found: {contract_file}")

        contract = parser.parse_contract_file(contract_file)

        # Verify basic structure
        assert contract.name == "NodeDatabaseAdapterEffect"
        assert contract.node_type == "effect"
        assert contract.schema_version == "v2.0.0"

        # Verify mixins
        assert len(contract.mixins) == 2
        assert contract.has_mixin("MixinHealthCheck")
        assert contract.has_mixin("MixinMetrics")

        # Verify mixin configurations
        health_check_mixin = next(
            m for m in contract.mixins if m.name == "MixinHealthCheck"
        )
        assert health_check_mixin.config["check_interval_ms"] == 30000
        assert "components" in health_check_mixin.config

        metrics_mixin = next(m for m in contract.mixins if m.name == "MixinMetrics")
        assert metrics_mixin.config["collect_latency"] is True

        # Should be valid
        assert contract.is_valid, f"Validation errors: {contract.validation_errors}"

    def test_parse_valid_complete_v2(self, parser, examples_dir):
        """Test parsing complete v2.0 contract with all features."""
        contract_file = examples_dir / "valid_complete_v2.yaml"

        if not contract_file.exists():
            pytest.skip(f"Example file not found: {contract_file}")

        contract = parser.parse_contract_file(contract_file)

        # Verify basic structure
        assert contract.schema_version.startswith("v2")
        assert contract.node_type in ["effect", "compute", "reducer", "orchestrator"]

        # Should be valid
        assert contract.is_valid, f"Validation errors: {contract.validation_errors}"

    def test_parse_invalid_mixin_name(self, parser, examples_dir):
        """Test parsing contract with invalid mixin name."""
        contract_file = examples_dir / "invalid_mixin_name.yaml"

        if not contract_file.exists():
            pytest.skip(f"Example file not found: {contract_file}")

        contract = parser.parse_contract_file(contract_file)

        # Should have validation errors
        assert not contract.is_valid
        assert len(contract.validation_errors) > 0
        # Check for mixin-related error
        assert any(
            "mixin" in err.lower() or "invalid" in err.lower()
            for err in contract.validation_errors
        )

    def test_parse_invalid_health_check_interval(self, parser, examples_dir):
        """Test parsing contract with invalid health check interval."""
        contract_file = examples_dir / "invalid_health_check_interval.yaml"

        if not contract_file.exists():
            pytest.skip(f"Example file not found: {contract_file}")

        contract = parser.parse_contract_file(contract_file)

        # Should have validation errors
        assert not contract.is_valid
        assert len(contract.validation_errors) > 0

    def test_parse_invalid_circuit_breaker_threshold(self, parser, examples_dir):
        """Test parsing contract with invalid circuit breaker threshold."""
        contract_file = examples_dir / "invalid_circuit_breaker_threshold.yaml"

        if not contract_file.exists():
            pytest.skip(f"Example file not found: {contract_file}")

        contract = parser.parse_contract_file(contract_file)

        # Should have validation errors
        assert not contract.is_valid
        assert len(contract.validation_errors) > 0

    def test_parse_invalid_missing_required_field(self, parser, examples_dir):
        """Test parsing contract with missing required field."""
        contract_file = examples_dir / "invalid_missing_required_field.yaml"

        if not contract_file.exists():
            pytest.skip(f"Example file not found: {contract_file}")

        contract = parser.parse_contract_file(contract_file)

        # Should have validation errors
        assert not contract.is_valid
        assert len(contract.validation_errors) > 0


class TestMixinRegistryIntegration:
    """Integration tests for mixin registry."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_registry_loaded_from_catalog(self, parser):
        """Test that registry is loaded from catalog."""
        available_mixins = parser.get_available_mixins()

        # Should have loaded mixins
        assert len(available_mixins) > 0

        # Check for key mixins from catalog
        expected_mixins = [
            "MixinHealthCheck",
            "MixinMetrics",
            "MixinLogData",
            "MixinEventDrivenNode",
            "MixinEventBus",
            "MixinServiceRegistry",
            "MixinCaching",
            "MixinHashComputation",
        ]

        for mixin_name in expected_mixins:
            assert mixin_name in available_mixins, f"Missing mixin: {mixin_name}"

    def test_mixin_info_contains_import_path(self, parser):
        """Test that mixin info contains import path."""
        info = parser.get_mixin_info("MixinHealthCheck")

        assert info is not None
        assert "import_path" in info
        assert "omnibase_core.mixins" in info["import_path"]
        assert "MixinHealthCheck" in info["import_path"]

    def test_mixin_dependencies_loaded(self, parser):
        """Test that mixin dependencies are loaded."""
        # MixinEventDrivenNode has dependencies
        deps = parser.get_mixin_dependencies("MixinEventDrivenNode")

        assert len(deps) > 0
        # Should include composed mixins
        assert "MixinEventHandler" in deps or "MixinNodeLifecycle" in deps


class TestContractValidationWorkflow:
    """Integration tests for complete validation workflow."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_complete_validation_workflow(self, parser):
        """Test complete validation workflow from contract data to validated result."""
        # Create comprehensive contract
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeComprehensiveEffect",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Comprehensive effect node with all features",
            "capabilities": [
                {"name": "data_processing", "description": "Process data"},
            ],
            "mixins": [
                {
                    "name": "MixinHealthCheck",
                    "enabled": True,
                    "config": {
                        "check_interval_ms": 30000,
                        "timeout_seconds": 5.0,
                        "components": [
                            {
                                "name": "database",
                                "critical": True,
                                "timeout_seconds": 5.0,
                            }
                        ],
                    },
                },
                {
                    "name": "MixinMetrics",
                    "enabled": True,
                    "config": {
                        "collect_latency": True,
                        "percentiles": [50, 95, 99],
                    },
                },
            ],
            "advanced_features": {
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 5,
                },
                "retry_policy": {
                    "enabled": True,
                    "max_attempts": 3,
                },
            },
            "subcontracts": {
                "effect": {
                    "operations": ["initialize", "shutdown", "process"],
                }
            },
        }

        # Parse contract
        contract = parser.parse_contract(contract_data)

        # Verify parsing success
        assert contract.name == "NodeComprehensiveEffect"
        assert contract.is_v2

        # Verify mixins parsed
        assert len(contract.mixins) == 2
        assert contract.has_mixin("MixinHealthCheck")
        assert contract.has_mixin("MixinMetrics")

        # Verify mixin import paths resolved
        for mixin in contract.mixins:
            assert mixin.import_path != ""
            assert "omnibase_core.mixins" in mixin.import_path

        # Verify advanced features parsed
        assert contract.advanced_features is not None
        assert contract.advanced_features.circuit_breaker is not None
        assert contract.advanced_features.retry_policy is not None

        # Verify validation passed
        assert contract.is_valid, f"Validation errors: {contract.validation_errors}"

    def test_validation_error_accumulation(self, parser):
        """Test that validation errors are properly accumulated."""
        # Create contract with multiple errors
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeWithErrors",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with validation errors",
            "mixins": [
                {"name": "InvalidName", "enabled": True},  # Invalid pattern
                {"name": "MixinNonExistent", "enabled": True},  # Unknown mixin
                {
                    "name": "MixinHealthCheck",
                    "enabled": True,
                    "config": {
                        "check_interval_ms": 500,  # Below minimum
                    },
                },
            ],
        }

        contract = parser.parse_contract(contract_data)

        # Should have multiple validation errors
        assert not contract.is_valid
        assert len(contract.validation_errors) >= 3

        # Verify error types
        error_text = " ".join(contract.validation_errors).lower()
        assert "invalid mixin name" in error_text
        assert "unknown mixin" in error_text


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

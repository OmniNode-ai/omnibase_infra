#!/usr/bin/env python3
"""
Unit tests for YAMLContractParser.

Tests:
- Parsing v1.0 contracts (backward compatibility)
- Parsing v2.0 contracts with mixins
- Mixin validation
- Dependency checking
- Configuration validation
- JSON Schema validation
- Error handling
"""

import pytest

from omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser


class TestYAMLContractParserBasic:
    """Test basic parsing functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_parser_initialization(self, parser):
        """Test parser initializes successfully."""
        assert parser is not None
        assert parser.contract_schema is not None
        assert parser.mixins_schema is not None
        assert parser.advanced_features_schema is not None

    def test_get_available_mixins(self, parser):
        """Test getting available mixins."""
        mixins = parser.get_available_mixins()
        assert len(mixins) > 0
        assert "MixinHealthCheck" in mixins
        assert "MixinMetrics" in mixins
        assert "MixinEventDrivenNode" in mixins

    def test_get_mixin_info(self, parser):
        """Test getting mixin info."""
        info = parser.get_mixin_info("MixinHealthCheck")
        assert info is not None
        assert "import_path" in info
        assert "omnibase_core.mixins.mixin_health_check" in info["import_path"]

    def test_get_mixin_dependencies(self, parser):
        """Test getting mixin dependencies."""
        # MixinEventDrivenNode has dependencies
        deps = parser.get_mixin_dependencies("MixinEventDrivenNode")
        assert len(deps) > 0
        assert "MixinEventHandler" in deps
        assert "MixinNodeLifecycle" in deps

        # MixinHealthCheck has no dependencies
        deps = parser.get_mixin_dependencies("MixinHealthCheck")
        assert len(deps) == 0


class TestV1ContractParsing:
    """Test parsing v1.0 contracts (backward compatibility)."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_parse_minimal_v1_contract(self, parser):
        """Test parsing minimal v1.0 contract."""
        contract_data = {
            "name": "NodeMinimalEffect",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Minimal effect node",
            "subcontracts": {"effect": {"operations": ["initialize", "shutdown"]}},
        }

        contract = parser.parse_contract(contract_data)

        assert contract.name == "NodeMinimalEffect"
        assert contract.version.major == 1
        assert contract.node_type == "effect"
        assert contract.schema_version == "v1.0.0"
        assert len(contract.mixins) == 0
        assert contract.advanced_features is None
        assert contract.is_valid

    def test_parse_v1_contract_with_capabilities(self, parser):
        """Test parsing v1.0 contract with capabilities."""
        contract_data = {
            "name": "NodeDatabaseEffect",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Database effect node",
            "capabilities": [
                {"name": "persistence", "description": "Persist data"},
            ],
            "subcontracts": {"effect": {"operations": ["process"]}},
        }

        contract = parser.parse_contract(contract_data)

        assert contract.name == "NodeDatabaseEffect"
        assert len(contract.capabilities) == 1
        assert contract.capabilities[0]["name"] == "persistence"
        assert contract.is_valid


class TestV2ContractParsing:
    """Test parsing v2.0 contracts with mixins."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_parse_contract_with_single_mixin(self, parser):
        """Test parsing contract with single mixin."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeWithHealthCheck",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with health check",
            "mixins": [
                {
                    "name": "MixinHealthCheck",
                    "enabled": True,
                    "config": {"check_interval_ms": 30000, "timeout_seconds": 5.0},
                }
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert contract.schema_version == "v2.0.0"
        assert contract.is_v2
        assert len(contract.mixins) == 1
        assert contract.mixins[0].name == "MixinHealthCheck"
        assert contract.mixins[0].enabled
        assert contract.mixins[0].config["check_interval_ms"] == 30000
        assert contract.is_valid

    def test_parse_contract_with_multiple_mixins(self, parser):
        """Test parsing contract with multiple mixins."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeWithMultipleMixins",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with multiple mixins",
            "mixins": [
                {"name": "MixinHealthCheck", "enabled": True},
                {"name": "MixinMetrics", "enabled": True},
                {"name": "MixinLogData", "enabled": True},
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert len(contract.mixins) == 3
        assert contract.get_mixin_names() == [
            "MixinHealthCheck",
            "MixinMetrics",
            "MixinLogData",
        ]
        assert contract.has_mixin("MixinHealthCheck")
        assert contract.has_mixin("MixinMetrics")
        assert contract.is_valid

    def test_parse_contract_with_disabled_mixin(self, parser):
        """Test parsing contract with disabled mixin."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeWithDisabledMixin",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with disabled mixin",
            "mixins": [
                {"name": "MixinHealthCheck", "enabled": True},
                {"name": "MixinMetrics", "enabled": False},
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert len(contract.mixins) == 2
        assert len(contract.get_enabled_mixins()) == 1
        assert contract.has_mixin("MixinHealthCheck")
        assert not contract.has_mixin("MixinMetrics")
        assert contract.is_valid


class TestMixinValidation:
    """Test mixin validation."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_invalid_mixin_name_pattern(self, parser):
        """Test validation of invalid mixin name pattern."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeInvalidMixin",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with invalid mixin name",
            "mixins": [
                {
                    "name": "InvalidName",
                    "enabled": True,
                },  # Doesn't follow Mixin* pattern
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert not contract.is_valid
        assert len(contract.validation_errors) > 0
        assert any("Invalid mixin name" in err for err in contract.validation_errors)

    def test_unknown_mixin_name(self, parser):
        """Test validation of unknown mixin name."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeUnknownMixin",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with unknown mixin",
            "mixins": [
                {"name": "MixinNonExistent", "enabled": True},
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert not contract.is_valid
        assert len(contract.validation_errors) > 0
        assert any("Unknown mixin" in err for err in contract.validation_errors)

    def test_mixin_config_validation(self, parser):
        """Test validation of mixin configuration."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeInvalidConfig",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with invalid mixin config",
            "mixins": [
                {
                    "name": "MixinHealthCheck",
                    "enabled": True,
                    "config": {
                        "check_interval_ms": 500,  # Below minimum of 1000
                    },
                }
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert not contract.is_valid
        assert len(contract.validation_errors) > 0
        # Check for config validation error
        assert any(
            "config validation failed" in err for err in contract.validation_errors
        )


class TestMixinDependencies:
    """Test mixin dependency checking."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_satisfied_dependencies(self, parser):
        """Test contract with satisfied mixin dependencies."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeSatisfiedDeps",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with satisfied dependencies",
            "mixins": [
                # MixinEventDrivenNode requires MixinEventHandler, MixinNodeLifecycle, MixinIntrospectionPublisher
                {"name": "MixinEventHandler", "enabled": True},
                {"name": "MixinNodeLifecycle", "enabled": True},
                {"name": "MixinIntrospectionPublisher", "enabled": True},
                {"name": "MixinEventDrivenNode", "enabled": True},
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert contract.is_valid

    def test_unsatisfied_dependencies(self, parser):
        """Test contract with unsatisfied mixin dependencies."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeUnsatisfiedDeps",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with unsatisfied dependencies",
            "mixins": [
                # MixinEventDrivenNode requires dependencies, but they're missing
                {"name": "MixinEventDrivenNode", "enabled": True},
            ],
        }

        contract = parser.parse_contract(contract_data)

        assert not contract.is_valid
        assert len(contract.validation_errors) > 0
        # Should have multiple dependency errors
        dep_errors = [
            err
            for err in contract.validation_errors
            if "dependency not satisfied" in err
        ]
        assert len(dep_errors) > 0


class TestAdvancedFeatures:
    """Test parsing advanced_features section."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_parse_circuit_breaker(self, parser):
        """Test parsing circuit breaker configuration."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeWithCircuitBreaker",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with circuit breaker",
            "advanced_features": {
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 5,
                    "recovery_timeout_ms": 60000,
                }
            },
        }

        contract = parser.parse_contract(contract_data)

        assert contract.advanced_features is not None
        assert contract.advanced_features.circuit_breaker is not None
        assert contract.advanced_features.circuit_breaker.enabled
        assert contract.advanced_features.circuit_breaker.failure_threshold == 5
        assert contract.is_valid

    def test_parse_retry_policy(self, parser):
        """Test parsing retry policy configuration."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeWithRetry",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with retry policy",
            "advanced_features": {
                "retry_policy": {
                    "enabled": True,
                    "max_attempts": 3,
                    "initial_delay_ms": 1000,
                    "backoff_multiplier": 2.0,
                }
            },
        }

        contract = parser.parse_contract(contract_data)

        assert contract.advanced_features is not None
        assert contract.advanced_features.retry_policy is not None
        assert contract.advanced_features.retry_policy.enabled
        assert contract.advanced_features.retry_policy.max_attempts == 3
        assert contract.is_valid

    def test_parse_multiple_advanced_features(self, parser):
        """Test parsing multiple advanced features."""
        contract_data = {
            "schema_version": "v2.0.0",
            "name": "NodeWithMultipleFeatures",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Node with multiple advanced features",
            "advanced_features": {
                "circuit_breaker": {"enabled": True, "failure_threshold": 5},
                "retry_policy": {"enabled": True, "max_attempts": 3},
                "dead_letter_queue": {"enabled": True, "max_retries": 3},
                "security_validation": {"enabled": True, "sanitize_inputs": True},
            },
        }

        contract = parser.parse_contract(contract_data)

        assert contract.advanced_features is not None
        assert contract.advanced_features.circuit_breaker is not None
        assert contract.advanced_features.retry_policy is not None
        assert contract.advanced_features.dead_letter_queue is not None
        assert contract.advanced_features.security_validation is not None
        assert contract.is_valid


class TestBackwardCompatibility:
    """Test backward compatibility with v1.0 contracts."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_v1_contract_without_schema_version(self, parser):
        """Test v1.0 contract without schema_version field."""
        contract_data = {
            "name": "NodeLegacy",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Legacy v1.0 contract",
        }

        contract = parser.parse_contract(contract_data)

        assert contract.schema_version == "v1.0.0"
        assert not contract.is_v2
        assert len(contract.mixins) == 0
        assert contract.is_valid

    def test_deprecated_error_handling_field(self, parser):
        """Test contract with deprecated error_handling field."""
        contract_data = {
            "name": "NodeDeprecated",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Contract with deprecated error_handling",
            "error_handling": {
                "retry_policy": {"max_attempts": 3},
            },
        }

        contract = parser.parse_contract(contract_data)

        assert contract.has_deprecated_error_handling()
        assert not contract.is_valid  # Should have deprecation warning
        assert any("DEPRECATED" in err for err in contract.validation_errors)


class TestErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return YAMLContractParser()

    def test_parse_invalid_yaml_structure(self, parser):
        """Test parsing contract with invalid structure."""
        # Missing required fields
        contract_data = {
            "name": "NodeMissingFields",
        }

        contract = parser.parse_contract(contract_data)

        # Should have validation errors for missing required fields
        assert not contract.is_valid

    def test_parse_empty_contract(self, parser):
        """Test parsing empty contract."""
        contract_data = {}

        contract = parser.parse_contract(contract_data)

        assert contract.name == ""
        assert not contract.is_valid


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

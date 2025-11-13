#!/usr/bin/env python3
"""
Integration tests for Phase 3 contract processing.

Tests the complete workflow:
1. Parse contract with Phase 3 fields
2. Validate contract with Phase 3 validation
3. Process subcontracts
"""

import pytest

from omninode_bridge.codegen.contracts import ContractValidator, SubcontractProcessor
from omninode_bridge.codegen.models_contract import (
    EnumFallbackStrategy,
    EnumLLMTier,
    EnumQualityLevel,
    EnumTemplateVariant,
    ModelEnhancedContract,
    ModelGenerationDirectives,
    ModelTemplateConfiguration,
    ModelVersionInfo,
)
from omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser


class TestPhase3ContractParsing:
    """Test Phase 3 contract parsing."""

    def test_parse_contract_with_template_config(self):
        """Test parsing contract with template configuration."""
        contract_data = {
            "name": "NodeTestEffect",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Test node with Phase 3 enhancements",
            "schema_version": "v2.1.0",
            "template": {
                "variant": "production",
                "patterns": ["circuit_breaker", "retry_policy"],
                "pattern_configuration": {"circuit_breaker": {"failure_threshold": 5}},
            },
        }

        parser = YAMLContractParser()
        contract = parser.parse_contract(contract_data)

        assert contract.name == "NodeTestEffect"
        assert contract.schema_version == "v2.1.0"
        assert contract.template.variant == EnumTemplateVariant.PRODUCTION
        assert "circuit_breaker" in contract.template.patterns
        assert contract.is_valid

    def test_parse_contract_with_generation_directives(self):
        """Test parsing contract with generation directives."""
        contract_data = {
            "name": "NodeTestCompute",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "compute",
            "description": "Test compute node",
            "schema_version": "v2.1.0",
            "generation": {
                "enable_llm": True,
                "llm_tier": "CLOUD_FAST",
                "quality_level": "production",
                "fallback_strategy": "graceful",
                "max_context_size": 8000,
            },
        }

        parser = YAMLContractParser()
        contract = parser.parse_contract(contract_data)

        assert contract.generation.enable_llm is True
        assert contract.generation.llm_tier == EnumLLMTier.CLOUD_FAST
        assert contract.generation.quality_level == EnumQualityLevel.PRODUCTION
        assert contract.generation.fallback_strategy == EnumFallbackStrategy.GRACEFUL
        assert contract.is_valid

    def test_parse_contract_with_quality_gates(self):
        """Test parsing contract with quality gates."""
        contract_data = {
            "name": "NodeTestOrchestrator",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "orchestrator",
            "description": "Test orchestrator node",
            "schema_version": "v2.1.0",
            "quality_gates": {
                "gates": [
                    {"gate": "syntax_validation", "required": True},
                    {"gate": "onex_compliance", "required": True},
                    {"gate": "security_scan", "required": False},
                ],
                "fail_on_first_error": True,
            },
        }

        parser = YAMLContractParser()
        contract = parser.parse_contract(contract_data)

        assert len(contract.quality_gates.gates) == 3
        assert contract.quality_gates.gates[0].name == "syntax_validation"
        assert contract.quality_gates.gates[0].required is True
        assert contract.is_valid


class TestPhase3ContractValidation:
    """Test Phase 3 contract validation."""

    def test_validate_valid_phase3_contract(self):
        """Test validation of valid Phase 3 contract."""
        contract = ModelEnhancedContract(
            name="NodeValidTest",
            version=ModelVersionInfo(major=1, minor=0, patch=0),
            node_type="effect",
            description="Valid test contract",
            schema_version="v2.1.0",
            template=ModelTemplateConfiguration(
                variant=EnumTemplateVariant.PRODUCTION,
                patterns=["circuit_breaker", "metrics"],
            ),
            generation=ModelGenerationDirectives(
                enable_llm=True,
                llm_tier=EnumLLMTier.CLOUD_FAST,
                quality_level=EnumQualityLevel.PRODUCTION,
            ),
        )

        validator = ContractValidator()
        result = validator.validate_contract(contract)

        assert result.is_valid
        assert result.error_count == 0

    def test_validate_invalid_pattern_names(self):
        """Test validation catches invalid pattern names."""
        contract = ModelEnhancedContract(
            name="NodeInvalidPatterns",
            version=ModelVersionInfo(major=1, minor=0, patch=0),
            node_type="effect",
            description="Contract with invalid patterns",
            schema_version="v2.1.0",
            template=ModelTemplateConfiguration(
                variant=EnumTemplateVariant.STANDARD,
                patterns=["invalid_pattern", "nonexistent_pattern"],
            ),
        )

        validator = ContractValidator()
        result = validator.validate_contract(contract)

        assert not result.is_valid
        assert result.error_count > 0
        assert any("invalid_pattern" in err.error_message for err in result.errors)

    def test_validate_custom_template_without_path(self):
        """Test validation catches custom template without path."""
        contract = ModelEnhancedContract(
            name="NodeCustomTemplate",
            version=ModelVersionInfo(major=1, minor=0, patch=0),
            node_type="effect",
            description="Contract with custom template",
            schema_version="v2.1.0",
            template=ModelTemplateConfiguration(
                variant=EnumTemplateVariant.CUSTOM, custom_template=None  # Missing path
            ),
        )

        validator = ContractValidator()
        result = validator.validate_contract(contract)

        assert not result.is_valid
        assert any("custom_template" in err.field_name for err in result.errors)


class TestSubcontractProcessing:
    """Test subcontract processing."""

    def test_process_database_subcontract(self):
        """Test processing database subcontract."""
        contract = ModelEnhancedContract(
            name="NodeDatabaseTest",
            version=ModelVersionInfo(major=1, minor=0, patch=0),
            node_type="effect",
            description="Test with database subcontract",
            schema_version="v2.1.0",
            subcontracts={
                "user_db": {
                    "type": "database",
                    "table_name": "users",
                    "operations": ["select", "insert", "update"],
                    "use_transactions": True,
                }
            },
        )

        processor = SubcontractProcessor()
        results = processor.process_subcontracts(contract)

        assert results.success_count == 1
        assert not results.has_errors
        assert len(results.all_imports) > 0
        assert any("DatabaseClient" in imp for imp in results.all_imports)

    def test_process_multiple_subcontracts(self):
        """Test processing multiple subcontracts."""
        contract = ModelEnhancedContract(
            name="NodeMultiSubcontract",
            version=ModelVersionInfo(major=1, minor=0, patch=0),
            node_type="orchestrator",
            description="Test with multiple subcontracts",
            schema_version="v2.1.0",
            subcontracts={
                "api_client": {
                    "type": "api",
                    "base_url": "http://api.example.com",
                    "endpoints": ["users", "posts"],
                    "use_retry": True,
                },
                "event_publisher": {
                    "type": "event",
                    "topics": ["user.created", "user.updated"],
                    "use_kafka": True,
                },
            },
        )

        processor = SubcontractProcessor()
        results = processor.process_subcontracts(contract)

        assert results.success_count == 2
        assert not results.has_errors
        assert len(results.all_imports) > 0

    def test_process_invalid_subcontract_type(self):
        """Test processing with invalid subcontract type."""
        contract = ModelEnhancedContract(
            name="NodeInvalidSubcontract",
            version=ModelVersionInfo(major=1, minor=0, patch=0),
            node_type="effect",
            description="Test with invalid subcontract",
            schema_version="v2.1.0",
            subcontracts={"invalid": {"type": "invalid_type"}},
        )

        processor = SubcontractProcessor()
        results = processor.process_subcontracts(contract)

        assert results.has_errors
        assert results.success_count == 0


class TestBackwardCompatibility:
    """Test backward compatibility with Phase 1/2 contracts."""

    def test_parse_v1_contract(self):
        """Test parsing v1.0 contract (backward compatibility)."""
        contract_data = {
            "name": "NodeLegacyEffect",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "Legacy v1.0 contract",
        }

        parser = YAMLContractParser()
        contract = parser.parse_contract(contract_data)

        assert contract.name == "NodeLegacyEffect"
        assert contract.schema_version == "v1.0.0"  # Default
        # Phase 3 fields should have defaults
        assert contract.template.variant == EnumTemplateVariant.STANDARD
        assert contract.generation.llm_tier == EnumLLMTier.CLOUD_FAST
        assert contract.is_valid

    def test_parse_v2_0_contract_without_phase3(self):
        """Test parsing v2.0 contract without Phase 3 fields."""
        contract_data = {
            "name": "NodeV2ContractEffect",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "effect",
            "description": "v2.0 contract with mixins",
            "schema_version": "v2.0.0",
            "mixins": [{"name": "MixinHealthCheck", "enabled": True}],
        }

        parser = YAMLContractParser()
        contract = parser.parse_contract(contract_data)

        assert contract.name == "NodeV2ContractEffect"
        assert contract.schema_version == "v2.0.0"
        assert len(contract.mixins) == 1
        # Phase 3 fields should have defaults
        assert contract.template.variant == EnumTemplateVariant.STANDARD
        assert contract.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Integration test for mixin-enhanced code generation.

Tests the full integration of Wave 2 components:
- YAMLContractParser
- MixinInjector
- NodeValidator
- CodeGenerationService
"""

# We'll mock omnibase_core imports to avoid dependency issues
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules["omnibase_core"] = MagicMock()
sys.modules["omnibase_core.nodes"] = MagicMock()
sys.modules["omnibase_core.nodes.node_effect"] = MagicMock()
sys.modules["omnibase_core.models"] = MagicMock()
sys.modules["omnibase_core.models.core"] = MagicMock()
sys.modules["omnibase_core.models.core.model_container"] = MagicMock()
sys.modules["omnibase_core.mixins"] = MagicMock()

from omninode_bridge.codegen.node_classifier import (
    EnumNodeType,
    ModelClassificationResult,
)
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.service import CodeGenerationService


class TestMixinIntegration:
    """Test mixin-enhanced code generation integration."""

    def test_service_initialization_with_mixin_components(self):
        """Test that service initializes with all Wave 2 components."""
        service = CodeGenerationService(
            enable_mixin_validation=True,
            enable_type_checking=False,
        )

        # Verify Wave 2 components are initialized
        assert service.yaml_contract_parser is not None
        assert service.mixin_injector is not None
        assert service.node_validator is not None
        assert service.enable_mixin_validation is True
        assert service.enable_type_checking is False

    def test_service_initialization_backward_compatible(self):
        """Test that service initialization is backward compatible."""
        # Should work without new parameters
        service = CodeGenerationService()

        # Should have default values
        assert service.yaml_contract_parser is not None
        assert service.mixin_injector is not None
        assert service.node_validator is not None
        assert service.enable_mixin_validation is True  # Default
        assert service.enable_type_checking is False  # Default

    @pytest.mark.asyncio
    async def test_generate_node_without_mixins_backward_compatible(self):
        """Test that generate_node works without mixins (backward compatible)."""
        service = CodeGenerationService()
        service._initialize_strategies()

        requirements = ModelPRDRequirements(
            service_name="test_service",
            node_type="effect",
            domain="test",
            business_description="Test service",
            operations=["test"],
            features=[],
            input_schema={},
            output_schema={},
            performance_requirements={},
            error_handling_strategy="retry",
            dependencies={},
        )

        classification = ModelClassificationResult(
            node_type=EnumNodeType.EFFECT,
            confidence=0.9,
            reasoning="Test classification",
        )

        # Mock the strategy generate method
        from uuid import uuid4

        from omninode_bridge.codegen.strategies.base import (
            EnumStrategyType,
            ModelGenerationResult,
        )
        from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts

        mock_artifacts = ModelGeneratedArtifacts(
            node_file="# Generated code",
            contract_file="# Contract YAML",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodeTestServiceEffect",
            service_name="test_service",
            models={},
            tests={},
            documentation={},
        )

        mock_result = ModelGenerationResult(
            artifacts=mock_artifacts,
            strategy_used=EnumStrategyType.JINJA2,
            generation_time_ms=500.0,
            validation_passed=True,
            validation_errors=[],
            llm_used=False,
            intelligence_sources=[],
            correlation_id=uuid4(),
        )

        # Patch strategy.generate
        with patch.object(
            service.strategy_registry.get_strategy(EnumStrategyType.JINJA2),
            "generate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Generate without mixins (backward compatible)
            result = await service.generate_node(
                requirements=requirements,
                classification=classification,
                output_directory=Path("/tmp/test"),
                strategy="jinja2",
                enable_llm=False,
                enable_mixins=False,  # Explicitly disable mixins
                validation_level="standard",
            )

            # Should succeed without errors
            assert result is not None
            assert result.artifacts.node_name == "NodeTestServiceEffect"
            assert result.validation_passed is True

    def test_enable_mixins_parameter_defaults_to_true(self):
        """Test that enable_mixins parameter defaults to True."""
        import inspect

        from omninode_bridge.codegen.service import CodeGenerationService

        sig = inspect.signature(CodeGenerationService.generate_node)
        enable_mixins_param = sig.parameters["enable_mixins"]

        assert enable_mixins_param.default is True


@pytest.mark.integration
class TestMixinSystemIntegration:
    """Integration tests for Phase 3 mixin system."""

    @pytest.mark.asyncio
    async def test_mixin_recommendation_and_injection(self):
        """Test mixin recommendation followed by injection."""
        # TODO: Test recommendation + injection integration
        pass

    @pytest.mark.asyncio
    async def test_mixin_conflict_detection_and_resolution(self):
        """Test conflict detection and resolution."""
        # TODO: Test conflict handling integration
        pass

    @pytest.mark.asyncio
    async def test_mixin_with_template_rendering(self):
        """Test mixin injection with template rendering."""
        # TODO: Test rendering integration
        pass


@pytest.mark.integration
class TestMixinCompatibility:
    """Integration tests for mixin compatibility."""

    @pytest.mark.asyncio
    async def test_multiple_mixins_compatibility(self):
        """Test compatibility of multiple mixins together."""
        # TODO: Test multi-mixin compatibility
        pass

    @pytest.mark.asyncio
    async def test_mixin_with_node_types(self):
        """Test mixin compatibility across node types."""
        # TODO: Test node type compatibility
        pass

    @pytest.mark.asyncio
    async def test_mixin_with_patterns(self):
        """Test mixin compatibility with patterns."""
        # TODO: Test pattern compatibility
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Unit tests for Jinja2Strategy.

Tests template-based code generation without LLM calls.
"""

from unittest.mock import AsyncMock, patch

import pytest

from omninode_bridge.codegen.node_classifier import EnumNodeType
from omninode_bridge.codegen.strategies.base import (
    EnumStrategyType,
    EnumValidationLevel,
    ModelGenerationRequest,
)
from omninode_bridge.codegen.strategies.jinja2_strategy import Jinja2Strategy
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts


class TestJinja2Strategy:
    """Test Jinja2Strategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create Jinja2Strategy instance."""
        return Jinja2Strategy(
            templates_directory=None,
            enable_inline_templates=True,
            enable_validation=True,
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.strategy_name == "Jinja2 Template Strategy"
        assert strategy.strategy_type == EnumStrategyType.JINJA2
        assert strategy.enable_validation is True
        assert strategy.template_engine is not None

    def test_supports_node_type_all_types(self, strategy):
        """Test that Jinja2 supports all standard node types."""
        assert strategy.supports_node_type(EnumNodeType.EFFECT) is True
        assert strategy.supports_node_type(EnumNodeType.COMPUTE) is True
        assert strategy.supports_node_type(EnumNodeType.REDUCER) is True
        assert strategy.supports_node_type(EnumNodeType.ORCHESTRATOR) is True

    def test_get_strategy_info(self, strategy):
        """Test getting strategy info."""
        info = strategy.get_strategy_info()

        # Verify structure
        assert info["name"] == "Jinja2 Template Strategy"
        assert info["type"] == "jinja2"
        assert info["requires_llm"] is False
        assert "performance_profile" in info
        assert info["performance_profile"]["cost_per_generation_usd"] == 0.0

        # Verify supported node types
        assert "effect" in info["supported_node_types"]
        assert "compute" in info["supported_node_types"]
        assert "reducer" in info["supported_node_types"]
        assert "orchestrator" in info["supported_node_types"]

    @pytest.mark.asyncio
    async def test_generate_simple_node(
        self, strategy, simple_crud_requirements, effect_classification, temp_output_dir
    ):
        """Test generating a simple CRUD node."""
        # Create mock artifacts
        mock_artifacts = ModelGeneratedArtifacts(
            node_file="# Generated Effect Node",
            contract_file="# Contract YAML",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodePostgresCrudEffect",
            service_name="postgres_crud",
            models={},
            tests={},
            documentation={},
        )

        # Mock template_engine.generate
        with patch.object(
            strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Create request
            request = ModelGenerationRequest(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
                strategy=EnumStrategyType.JINJA2,
                enable_llm=False,
                validation_level=EnumValidationLevel.STANDARD,
            )

            # Generate
            result = await strategy.generate(request)

            # Verify result
            assert result is not None
            assert result.strategy_used == EnumStrategyType.JINJA2
            assert result.llm_used is False
            assert result.validation_passed is True
            assert result.artifacts.node_name == "NodePostgresCrudEffect"
            assert result.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_validates_requirements(
        self, strategy, invalid_requirements, effect_classification, temp_output_dir
    ):
        """Test that generation validates requirements."""
        request = ModelGenerationRequest(
            requirements=invalid_requirements,
            classification=effect_classification,
            output_directory=temp_output_dir,
            validation_level=EnumValidationLevel.STRICT,
        )

        # Should raise ValueError due to validation failure
        with pytest.raises(ValueError, match="Requirements validation failed"):
            await strategy.generate(request)

    @pytest.mark.asyncio
    async def test_generate_handles_template_error(
        self, strategy, simple_crud_requirements, effect_classification, temp_output_dir
    ):
        """Test that generation handles template engine errors."""
        # Mock template_engine.generate to raise error
        with patch.object(
            strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            side_effect=Exception("Template not found"),
        ):
            request = ModelGenerationRequest(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
            )

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Jinja2 code generation failed"):
                await strategy.generate(request)

    @pytest.mark.asyncio
    async def test_generate_tracks_generation_time(
        self, strategy, simple_crud_requirements, effect_classification, temp_output_dir
    ):
        """Test that generation tracks execution time."""
        mock_artifacts = ModelGeneratedArtifacts(
            node_file="# Generated code",
            contract_file="# Contract",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodePostgresCrudEffect",
            service_name="postgres_crud",
            models={},
            tests={},
            documentation={},
        )

        with patch.object(
            strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            request = ModelGenerationRequest(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
            )

            result = await strategy.generate(request)

            # Verify generation time is tracked
            assert result.generation_time_ms >= 0
            # Should be fast for mocked template engine
            assert result.generation_time_ms < 5000  # < 5 seconds

    @pytest.mark.asyncio
    async def test_generate_validation_disabled(
        self, simple_crud_requirements, effect_classification, temp_output_dir
    ):
        """Test generation with validation disabled."""
        # Create strategy with validation disabled
        strategy = Jinja2Strategy(
            templates_directory=None,
            enable_inline_templates=True,
            enable_validation=False,
        )

        mock_artifacts = ModelGeneratedArtifacts(
            node_file="# Generated code",
            contract_file="# Contract",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodePostgresCrudEffect",
            service_name="postgres_crud",
            models={},
            tests={},
            documentation={},
        )

        with patch.object(
            strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Use invalid requirements (should still work with validation disabled)
            from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

            invalid_reqs = ModelPRDRequirements(
                service_name="",  # Empty service name
                node_type="effect",
                domain="",
                business_description="",
                operations=[],
                features=[],
                input_schema={},
                output_schema={},
                performance_requirements={},
                error_handling_strategy="",
                dependencies={},
            )

            request = ModelGenerationRequest(
                requirements=invalid_reqs,
                classification=effect_classification,
                output_directory=temp_output_dir,
                validation_level=EnumValidationLevel.NONE,
            )

            # Should not raise error with validation disabled
            result = await strategy.generate(request)
            assert result is not None

    def test_validate_requirements_basic_level(
        self, strategy, simple_crud_requirements
    ):
        """Test requirements validation at basic level."""
        is_valid, errors = strategy.validate_requirements(
            simple_crud_requirements, EnumValidationLevel.BASIC
        )

        # Simple requirements should pass basic validation
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_requirements_strict_level(
        self, strategy, simple_crud_requirements
    ):
        """Test requirements validation at strict level."""
        is_valid, errors = strategy.validate_requirements(
            simple_crud_requirements, EnumValidationLevel.STRICT
        )

        # Simple requirements should pass strict validation
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_requirements_missing_service_name(self, strategy):
        """Test validation catches missing service name."""
        from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

        invalid_reqs = ModelPRDRequirements(
            service_name="",  # Empty
            node_type="effect",
            domain="test",
            business_description="Test",
            operations=["test"],
            features=[],
            input_schema={},
            output_schema={},
            performance_requirements={},
            error_handling_strategy="retry",
            dependencies={},
        )

        is_valid, errors = strategy.validate_requirements(
            invalid_reqs, EnumValidationLevel.BASIC
        )

        assert is_valid is False
        assert "service_name is required" in errors

    def test_validate_requirements_missing_operations_strict(self, strategy):
        """Test strict validation catches missing operations."""
        from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

        reqs_without_ops = ModelPRDRequirements(
            service_name="test_service",
            node_type="effect",
            domain="test",
            business_description="Test service",
            operations=[],  # Empty operations
            features=[],
            input_schema={},
            output_schema={},
            performance_requirements={},
            error_handling_strategy="retry",
            dependencies={},
        )

        is_valid, errors = strategy.validate_requirements(
            reqs_without_ops, EnumValidationLevel.STRICT
        )

        assert is_valid is False
        assert "At least one operation is required" in errors

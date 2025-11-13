#!/usr/bin/env python3
"""
Unit tests for CodeGenerationService.

Tests the facade layer that orchestrates code generation across all strategies.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omninode_bridge.codegen.node_classifier import EnumNodeType
from omninode_bridge.codegen.service import CodeGenerationService, StrategyRegistry
from omninode_bridge.codegen.strategies.base import (
    EnumStrategyType,
    EnumValidationLevel,
    ModelGenerationRequest,
    ModelGenerationResult,
)
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts


class TestStrategyRegistry:
    """Test StrategyRegistry functionality."""

    def test_register_strategy(self):
        """Test registering a strategy."""
        registry = StrategyRegistry()

        # Create mock strategy
        mock_strategy = MagicMock()
        mock_strategy.strategy_type = EnumStrategyType.JINJA2
        mock_strategy.strategy_name = "Test Strategy"

        # Register
        registry.register(mock_strategy)

        # Verify registration
        assert registry.get_strategy(EnumStrategyType.JINJA2) == mock_strategy

    def test_register_duplicate_strategy_raises_error(self):
        """Test that registering duplicate strategy raises error."""
        registry = StrategyRegistry()

        # Create mock strategies
        mock_strategy1 = MagicMock()
        mock_strategy1.strategy_type = EnumStrategyType.JINJA2
        mock_strategy1.strategy_name = "Strategy 1"

        mock_strategy2 = MagicMock()
        mock_strategy2.strategy_type = EnumStrategyType.JINJA2
        mock_strategy2.strategy_name = "Strategy 2"

        # Register first strategy
        registry.register(mock_strategy1)

        # Attempt to register duplicate should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            registry.register(mock_strategy2)

    def test_get_default_strategy(self):
        """Test getting default strategy."""
        registry = StrategyRegistry()

        # Create mock strategy
        mock_strategy = MagicMock()
        mock_strategy.strategy_type = EnumStrategyType.JINJA2
        mock_strategy.strategy_name = "Default Strategy"

        # Register as default
        registry.register(mock_strategy, is_default=True)

        # Verify default
        assert registry.get_default_strategy() == mock_strategy
        assert registry._default_strategy == EnumStrategyType.JINJA2

    def test_list_strategies(self):
        """Test listing all registered strategies."""
        registry = StrategyRegistry()

        # Create mock strategies
        jinja2_strategy = MagicMock()
        jinja2_strategy.strategy_type = EnumStrategyType.JINJA2
        jinja2_strategy.strategy_name = "Jinja2 Strategy"

        template_load_strategy = MagicMock()
        template_load_strategy.strategy_type = EnumStrategyType.TEMPLATE_LOADING
        template_load_strategy.strategy_name = "Template Load Strategy"

        # Register
        registry.register(jinja2_strategy, is_default=True)
        registry.register(template_load_strategy)

        # List strategies
        strategies = registry.list_strategies()

        # Verify
        assert len(strategies) == 2
        assert any(s["type"] == "jinja2" and s["is_default"] for s in strategies)
        assert any(
            s["type"] == "template_loading" and not s["is_default"] for s in strategies
        )

    def test_select_strategy_by_preference(self):
        """Test selecting strategy by preference."""
        registry = StrategyRegistry()

        # Create mock strategies
        jinja2_strategy = MagicMock()
        jinja2_strategy.strategy_type = EnumStrategyType.JINJA2
        jinja2_strategy.strategy_name = "Jinja2 Strategy"
        jinja2_strategy.supports_node_type.return_value = True

        # Register
        registry.register(jinja2_strategy, is_default=True)

        # Select with preference
        selected = registry.select_strategy(
            node_type=EnumNodeType.EFFECT,
            enable_llm=False,
            prefer_strategy=EnumStrategyType.JINJA2,
        )

        # Verify
        assert selected == jinja2_strategy

    def test_select_strategy_llm_enabled(self):
        """Test strategy selection when LLM is enabled."""
        registry = StrategyRegistry()

        # Create mock strategies
        jinja2_strategy = MagicMock()
        jinja2_strategy.strategy_type = EnumStrategyType.JINJA2
        jinja2_strategy.supports_node_type.return_value = True

        template_load_strategy = MagicMock()
        template_load_strategy.strategy_type = EnumStrategyType.TEMPLATE_LOADING
        template_load_strategy.supports_node_type.return_value = True

        # Register
        registry.register(jinja2_strategy, is_default=True)
        registry.register(template_load_strategy)

        # Select with LLM enabled
        selected = registry.select_strategy(
            node_type=EnumNodeType.EFFECT, enable_llm=True
        )

        # Should select TemplateLoad strategy when LLM is enabled
        assert selected == template_load_strategy

    def test_select_strategy_no_suitable_strategy_raises_error(self):
        """Test that selecting with no suitable strategy raises error."""
        registry = StrategyRegistry()

        # Create mock strategy that doesn't support the node type
        mock_strategy = MagicMock()
        mock_strategy.strategy_type = EnumStrategyType.JINJA2
        mock_strategy.supports_node_type.return_value = False

        registry.register(mock_strategy)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No suitable strategy found"):
            registry.select_strategy(node_type=EnumNodeType.EFFECT, enable_llm=False)


class TestCodeGenerationService:
    """Test CodeGenerationService functionality."""

    @pytest.fixture
    def service(self):
        """Create CodeGenerationService instance."""
        return CodeGenerationService(
            templates_directory=None,
            archon_mcp_url=None,
            enable_intelligence=False,  # Disable for unit tests
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service.node_classifier is not None
        assert service.strategy_registry is not None
        assert service._strategies_initialized is False

    @pytest.mark.asyncio
    async def test_generate_node_with_auto_strategy(
        self, service, simple_crud_requirements, effect_classification, temp_output_dir
    ):
        """Test generating node with AUTO strategy selection."""
        # Initialize strategies
        service._initialize_strategies()

        # Mock the Jinja2Strategy generate method
        mock_artifacts = ModelGeneratedArtifacts(
            node_file="# Generated code",
            contract_file="# Contract YAML",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodePostgresCrudEffect",
            service_name="postgres_crud",
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

        # Patch the strategy's generate method
        with patch.object(
            service.strategy_registry.get_strategy(EnumStrategyType.JINJA2),
            "generate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Generate node
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
                strategy="auto",
                enable_llm=False,
                validation_level="standard",
            )

            # Verify result
            assert result is not None
            assert result.strategy_used == EnumStrategyType.JINJA2
            assert result.validation_passed is True
            assert result.artifacts.node_name == "NodePostgresCrudEffect"

    @pytest.mark.asyncio
    async def test_generate_node_validates_requirements(
        self, service, invalid_requirements, effect_classification, temp_output_dir
    ):
        """Test that generating with invalid requirements fails validation."""
        service._initialize_strategies()

        # Should raise ValueError due to validation failure
        with pytest.raises(ValueError, match="Requirements validation failed"):
            await service.generate_node(
                requirements=invalid_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
                strategy="jinja2",
                enable_llm=False,
                validation_level="strict",
            )

    @pytest.mark.asyncio
    async def test_generate_node_auto_classifies_node_type(
        self, service, simple_crud_requirements, temp_output_dir
    ):
        """Test that node type is auto-classified when not provided."""
        service._initialize_strategies()

        # Mock artifacts
        mock_artifacts = ModelGeneratedArtifacts(
            node_file="# Generated code",
            contract_file="# Contract YAML",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodePostgresCrudEffect",
            service_name="postgres_crud",
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

        with patch.object(
            service.strategy_registry.get_strategy(EnumStrategyType.JINJA2),
            "generate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Generate without providing classification
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=None,  # Will auto-classify
                output_directory=temp_output_dir,
                strategy="jinja2",
                enable_llm=False,
            )

            # Verify node was classified and generated
            assert result is not None
            assert result.artifacts.node_name == "NodePostgresCrudEffect"

    def test_parse_strategy_type_valid(self, service):
        """Test parsing valid strategy type."""
        strategy_type = service._parse_strategy_type("jinja2")
        assert strategy_type == EnumStrategyType.JINJA2

        strategy_type = service._parse_strategy_type("template_loading")
        assert strategy_type == EnumStrategyType.TEMPLATE_LOADING

    def test_parse_strategy_type_invalid_raises_error(self, service):
        """Test that parsing invalid strategy type raises error."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            service._parse_strategy_type("invalid_strategy")

    def test_parse_validation_level_valid(self, service):
        """Test parsing valid validation level."""
        level = service._parse_validation_level("strict")
        assert level == EnumValidationLevel.STRICT

        level = service._parse_validation_level("standard")
        assert level == EnumValidationLevel.STANDARD

    def test_parse_validation_level_invalid_raises_error(self, service):
        """Test that parsing invalid validation level raises error."""
        with pytest.raises(ValueError, match="Invalid validation_level"):
            service._parse_validation_level("invalid_level")

    def test_validate_request_valid(
        self, service, simple_crud_requirements, effect_classification, temp_output_dir
    ):
        """Test validating a valid request."""
        request = ModelGenerationRequest(
            requirements=simple_crud_requirements,
            classification=effect_classification,
            output_directory=temp_output_dir,
            strategy=EnumStrategyType.JINJA2,
            enable_llm=False,
            validation_level=EnumValidationLevel.STANDARD,
        )

        is_valid, errors = service._validate_request(request)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_request_missing_service_name(
        self, service, effect_classification, temp_output_dir
    ):
        """Test validating request with missing service name."""
        from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

        invalid_requirements = ModelPRDRequirements(
            service_name="",  # Empty service name
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

        request = ModelGenerationRequest(
            requirements=invalid_requirements,
            classification=effect_classification,
            output_directory=temp_output_dir,
        )

        is_valid, errors = service._validate_request(request)

        assert is_valid is False
        assert "service_name is required" in errors

    def test_list_strategies(self, service):
        """Test listing registered strategies."""
        service._initialize_strategies()

        strategies = service.list_strategies()

        # Should have at least Jinja2Strategy registered
        assert len(strategies) >= 1
        assert any(s["type"] == "jinja2" for s in strategies)

    def test_get_strategy_info(self, service):
        """Test getting strategy info."""
        service._initialize_strategies()

        info = service.get_strategy_info("jinja2")

        # Verify info structure
        assert info["name"] is not None
        assert info["type"] == "jinja2"
        assert "supported_node_types" in info
        assert "performance_profile" in info

    def test_get_strategy_info_invalid_strategy_raises_error(self, service):
        """Test that getting info for invalid strategy raises error."""
        service._initialize_strategies()

        with pytest.raises(ValueError, match="Invalid strategy .* Valid options:"):
            service.get_strategy_info("invalid_strategy")

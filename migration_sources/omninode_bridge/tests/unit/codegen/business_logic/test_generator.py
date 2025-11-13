#!/usr/bin/env python3
"""
Unit tests for BusinessLogicGenerator.

Tests LLM-based business logic generation with mocked dependencies.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError

from omninode_bridge.codegen.business_logic import (
    BusinessLogicConfig,
    BusinessLogicGenerator,
    ModelBusinessLogicContext,
    ModelGeneratedMethod,
    ModelMethodStub,
)
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts
from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import EnumLLMTier


class TestBusinessLogicGeneratorInit:
    """Test BusinessLogicGenerator initialization."""

    def test_init_llm_disabled(self):
        """Test initialization with LLM disabled."""
        generator = BusinessLogicGenerator(enable_llm=False)

        assert generator.enable_llm is False
        assert generator.llm_node is None
        assert generator.llm_tier == EnumLLMTier.CLOUD_FAST

    @patch.dict(os.environ, {"ZAI_API_KEY": "test_key"})
    def test_init_llm_enabled(self):
        """Test initialization with LLM enabled."""
        generator = BusinessLogicGenerator(enable_llm=True)

        assert generator.enable_llm is True
        assert generator.llm_node is not None
        assert generator.llm_tier == EnumLLMTier.CLOUD_FAST

    def test_init_llm_enabled_no_api_key(self):
        """Test initialization fails without API key."""
        # Clear ZAI_API_KEY if set
        os.environ.pop("ZAI_API_KEY", None)

        with pytest.raises(ModelOnexError) as exc_info:
            BusinessLogicGenerator(enable_llm=True)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_INPUT
        assert "ZAI_API_KEY" in str(exc_info.value)

    def test_init_custom_tier(self):
        """Test initialization with custom LLM tier."""
        generator = BusinessLogicGenerator(
            enable_llm=False, llm_tier=EnumLLMTier.CLOUD_PREMIUM
        )

        assert generator.llm_tier == EnumLLMTier.CLOUD_PREMIUM


class TestExtractMethodStubs:
    """Test method stub extraction."""

    def test_extract_stubs_with_implementation_required(self):
        """Test extracting stubs with IMPLEMENTATION REQUIRED marker."""
        node_file = '''
class NodeTestEffect:
    async def execute_effect(self, contract):
        """Execute the effect."""
        # IMPLEMENTATION REQUIRED
        pass
'''

        generator = BusinessLogicGenerator(enable_llm=False)
        stubs = generator._extract_method_stubs(node_file, "effect")

        assert len(stubs) == 1
        assert stubs[0].method_name == "execute_effect"
        assert stubs[0].needs_implementation is True

    def test_extract_stubs_with_todo(self):
        """Test extracting stubs with TODO marker."""
        node_file = '''
class NodeTestEffect:
    async def process_data(self, data):
        """Process data."""
        # TODO: Implement data processing
        pass
'''

        generator = BusinessLogicGenerator(enable_llm=False)
        stubs = generator._extract_method_stubs(node_file, "effect")

        assert len(stubs) == 1
        assert stubs[0].method_name == "process_data"

    def test_extract_stubs_no_stubs(self):
        """Test extraction with no stubs present."""
        node_file = '''
class NodeTestEffect:
    async def execute_effect(self, contract):
        """Execute the effect."""
        return {"status": "success"}
'''

        generator = BusinessLogicGenerator(enable_llm=False)
        stubs = generator._extract_method_stubs(node_file, "effect")

        assert len(stubs) == 0

    def test_extract_stubs_syntax_error(self):
        """Test extraction handles syntax errors gracefully."""
        node_file = "this is not valid python code"

        generator = BusinessLogicGenerator(enable_llm=False)
        stubs = generator._extract_method_stubs(node_file, "effect")

        assert len(stubs) == 0


class TestBuildMethodContext:
    """Test method context building."""

    def test_build_method_context(self):
        """Test building context for method generation."""
        generator = BusinessLogicGenerator(enable_llm=False)

        stub = ModelMethodStub(
            method_name="execute_effect",
            signature="async def execute_effect(self, contract):",
            docstring="Execute the effect.",
            line_number=10,
            needs_implementation=True,
        )

        requirements = ModelPRDRequirements(
            node_type="effect",
            service_name="test_service",
            domain="database",
            business_description="Test service description",
            operations=["create", "read"],
            features=["caching", "retry"],
            performance_requirements={"latency_ms": 100},
        )

        context_data = {
            "patterns": ["Pattern 1", "Pattern 2"],
            "best_practices": ["Practice 1"],
        }

        context = generator._build_method_context(
            stub=stub,
            requirements=requirements,
            node_type="effect",
            context_data=context_data,
        )

        assert context.method_name == "execute_effect"
        assert context.service_name == "test_service"
        assert context.node_type == "effect"
        assert len(context.similar_patterns) == 2
        assert len(context.best_practices) == 1


class TestBuildGenerationPrompt:
    """Test prompt building."""

    def test_build_generation_prompt(self):
        """Test building structured prompt."""
        generator = BusinessLogicGenerator(enable_llm=False)

        context = ModelBusinessLogicContext(
            node_type="effect",
            service_name="test_service",
            business_description="Test description",
            operations=["create", "read"],
            features=["caching"],
            method_name="execute_effect",
            method_signature="async def execute_effect(self, contract):",
            method_docstring="Execute the effect.",
            similar_patterns=["Pattern 1"],
            best_practices=["Practice 1"],
            performance_requirements={"latency_ms": 100},
        )

        prompt = generator._build_generation_prompt(context)

        assert "execute_effect" in prompt
        assert "test_service" in prompt
        assert "effect" in prompt
        assert "Pattern 1" in prompt
        assert "Practice 1" in prompt
        assert "ModelOnexError" in prompt
        assert "emit_log_event" in prompt


class TestValidateGeneratedCode:
    """Test code validation."""

    def test_validate_valid_code(self):
        """Test validation of valid Python code."""
        generator = BusinessLogicGenerator(enable_llm=False)

        code = """
try:
    result: str = "test"
    emit_log_event(LogLevel.INFO, "Processing")
    return result
except Exception as e:
    raise ModelOnexError(error_code=EnumCoreErrorCode.EXECUTION_ERROR, message=str(e))
"""

        validation = generator._validate_generated_code(code)

        assert validation["syntax_valid"] is True
        assert validation["onex_compliant"] is True
        assert validation["has_type_hints"] is True
        assert len(validation["security_issues"]) == 0

    def test_validate_syntax_error(self):
        """Test validation detects syntax errors."""
        generator = BusinessLogicGenerator(enable_llm=False)

        code = "this is not valid python"

        validation = generator._validate_generated_code(code)

        assert validation["syntax_valid"] is False

    def test_validate_hardcoded_secrets(self):
        """Test validation detects hardcoded secrets."""
        generator = BusinessLogicGenerator(enable_llm=False)

        code = """
api_key = "sk-1234567890"
password = "secret123"
"""

        validation = generator._validate_generated_code(code)

        assert validation["syntax_valid"] is True
        assert len(validation["security_issues"]) > 0


@pytest.mark.asyncio
class TestEnhanceArtifacts:
    """Test artifact enhancement."""

    async def test_enhance_artifacts_llm_disabled(self):
        """Test enhancement with LLM disabled."""
        generator = BusinessLogicGenerator(enable_llm=False)

        artifacts = ModelGeneratedArtifacts(
            node_file="# node file",
            contract_file="# contract",
            init_file="# init",
            node_type="effect",
            node_name="NodeTestEffect",
            service_name="test_service",
        )

        requirements = ModelPRDRequirements(
            node_type="effect",
            service_name="test_service",
            domain="database",
            business_description="Test",
            operations=["create"],
            features=["caching"],
            performance_requirements={},
        )

        enhanced = await generator.enhance_artifacts(
            artifacts=artifacts, requirements=requirements
        )

        assert enhanced.enhanced_node_file == artifacts.node_file
        assert len(enhanced.methods_generated) == 0
        assert enhanced.generation_success_rate == 1.0

    @patch.dict(os.environ, {"ZAI_API_KEY": "test_key"})
    async def test_enhance_artifacts_llm_enabled(self):
        """Test enhancement with LLM enabled (mocked)."""
        generator = BusinessLogicGenerator(enable_llm=True)

        # Mock the LLM generation method directly
        async def mock_generate(context):
            return ModelGeneratedMethod(
                method_name=context.method_name,
                generated_code="        # Generated implementation\n        emit_log_event(LogLevel.INFO, 'test')\n        raise ModelOnexError(error_code=EnumCoreErrorCode.EXECUTION_ERROR, message='test')",
                syntax_valid=True,
                onex_compliant=True,
                has_type_hints=True,
                has_docstring=False,
                security_issues=[],
                tokens_used=150,
                cost_usd=0.001,
                latency_ms=1000.0,
                model_used="glm-4.5",
            )

        generator._generate_method_implementation = mock_generate

        artifacts = ModelGeneratedArtifacts(
            node_file='''
class NodeTestEffect:
    async def execute_effect(self, contract):
        """Execute effect."""
        # IMPLEMENTATION REQUIRED
        pass
''',
            contract_file="# contract",
            init_file="# init",
            node_type="effect",
            node_name="NodeTestEffect",
            service_name="test_service",
        )

        requirements = ModelPRDRequirements(
            node_type="effect",
            service_name="test_service",
            domain="database",
            business_description="Test",
            operations=["create"],
            features=["caching"],
            performance_requirements={},
        )

        enhanced = await generator.enhance_artifacts(
            artifacts=artifacts, requirements=requirements
        )

        assert len(enhanced.methods_generated) == 1
        assert enhanced.total_tokens_used == 150
        assert enhanced.total_cost_usd == 0.001

    async def test_enhance_artifacts_error_handling(self):
        """Test enhancement handles errors gracefully."""
        generator = BusinessLogicGenerator(enable_llm=False)

        # Invalid artifacts
        with pytest.raises(Exception):
            await generator.enhance_artifacts(
                artifacts=None,  # type: ignore
                requirements=None,  # type: ignore
            )


class TestCleanup:
    """Test resource cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_llm_disabled(self):
        """Test cleanup with LLM disabled."""
        generator = BusinessLogicGenerator(enable_llm=False)

        await generator.cleanup()  # Should not raise

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"ZAI_API_KEY": "test_key"})
    async def test_cleanup_llm_enabled(self):
        """Test cleanup with LLM enabled."""
        generator = BusinessLogicGenerator(enable_llm=True)
        generator.llm_node.cleanup = AsyncMock()

        await generator.cleanup()

        generator.llm_node.cleanup.assert_called_once()


class TestBusinessLogicConfig:
    """Test configuration."""

    def test_config_defaults(self):
        """Test configuration default values."""
        assert BusinessLogicConfig.DEFAULT_LLM_TIER == EnumLLMTier.CLOUD_FAST
        assert BusinessLogicConfig.DEFAULT_TEMPERATURE == 0.3
        assert BusinessLogicConfig.DEFAULT_MAX_TOKENS == 2000
        assert len(BusinessLogicConfig.STUB_INDICATORS) > 0
        assert len(BusinessLogicConfig.ONEX_PATTERNS) > 0
        assert len(BusinessLogicConfig.SECURITY_PATTERNS) > 0

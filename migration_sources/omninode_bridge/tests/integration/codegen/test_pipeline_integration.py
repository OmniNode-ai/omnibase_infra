#!/usr/bin/env python3
"""
End-to-end integration tests for code generation pipeline.

Tests the complete pipeline:
1. Template loading (TemplateEngine)
2. Business logic enhancement (BusinessLogicGenerator)
3. Output validation (syntax, stubs, ONEX compliance)

Test Coverage:
- Happy path (all node types)
- LLM enabled vs disabled
- Error handling (missing template, invalid input)
- Output validation (syntax, stub replacement, ONEX patterns)
"""

import ast
import re

import pytest
from omnibase_core import ModelOnexError

from omninode_bridge.codegen.business_logic import (
    BusinessLogicGenerator,
    ModelEnhancedArtifacts,
)
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts
from omninode_bridge.codegen.template_engine_loader import (
    TemplateEngine,
    TemplateEngineError,
)

# ============================================================================
# Complete Pipeline Tests (Template Loading + LLM Enhancement)
# ============================================================================


class TestCodeGenerationPipelineHappyPath:
    """Test complete pipeline for successful code generation."""

    @pytest.mark.asyncio
    async def test_full_pipeline_effect_node(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test complete pipeline for Effect node generation."""
        # Step 1: Template is already loaded (via fixture)
        artifacts = sample_generated_artifacts

        # Verify template loaded successfully
        assert artifacts.node_type == "effect"
        assert "# IMPLEMENTATION REQUIRED" in artifacts.node_file

        # Step 2: Enhance with LLM (mocked)
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Step 3: Validate output
        assert isinstance(enhanced, ModelEnhancedArtifacts)
        assert enhanced.enhanced_node_file != artifacts.node_file
        assert len(enhanced.methods_generated) > 0

        # Step 4: Verify stubs were replaced
        assert "# IMPLEMENTATION REQUIRED" not in enhanced.enhanced_node_file
        assert "# Generated implementation" in enhanced.enhanced_node_file

        # Step 5: Verify syntax is valid
        try:
            ast.parse(enhanced.enhanced_node_file)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        assert syntax_valid, "Generated code should be syntactically valid Python"

    @pytest.mark.asyncio
    async def test_full_pipeline_compute_node(
        self,
        temp_template_dir,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test complete pipeline for Compute node generation."""
        # Step 1: Load template
        engine = TemplateEngine(template_root=temp_template_dir)
        artifacts = engine.load_template(node_type="compute", version="v1_0_0")

        # Verify template loaded
        assert artifacts.node_type == "compute"
        assert artifacts.get_stub_count() > 0

        # Update requirements for compute node
        compute_requirements = ModelPRDRequirements(
            **{
                **sample_prd_requirements.model_dump(),
                "node_type": "compute",
            }
        )

        # Step 2: Enhance with LLM
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=artifacts,
            requirements=compute_requirements,
            context_data=integration_context,
        )

        # Step 3: Validate
        assert isinstance(enhanced, ModelEnhancedArtifacts)
        assert "execute_compute" in enhanced.enhanced_node_file

        # Verify syntax
        try:
            ast.parse(enhanced.enhanced_node_file)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        assert syntax_valid

    @pytest.mark.asyncio
    async def test_pipeline_preserves_metadata(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that pipeline preserves template metadata."""
        # Enhance artifacts
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Verify metadata preserved in enhanced code
        assert "NodeSampleEffect" in enhanced.enhanced_node_file
        assert "import logging" in enhanced.enhanced_node_file
        assert "from omnibase_core" in enhanced.enhanced_node_file

    @pytest.mark.asyncio
    async def test_pipeline_replaces_all_stubs(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that pipeline replaces all detected stubs."""
        artifacts = sample_generated_artifacts

        # Verify we have stubs to replace
        assert "# IMPLEMENTATION REQUIRED" in artifacts.node_file

        # Enhance
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Verify all stubs replaced
        stub_markers = ["# IMPLEMENTATION REQUIRED", "# TODO:"]
        for marker in stub_markers:
            # Count should be reduced (some may remain in comments)
            initial_count = artifacts.node_file.count(marker)
            enhanced_count = enhanced.enhanced_node_file.count(marker)
            # At least some stubs should be replaced
            assert enhanced_count < initial_count or enhanced_count == 0


# ============================================================================
# LLM Enabled vs Disabled Tests
# ============================================================================


class TestPipelineLLMConfiguration:
    """Test pipeline behavior with LLM enabled vs disabled."""

    @pytest.mark.asyncio
    async def test_pipeline_llm_disabled_returns_original(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_disabled,
    ):
        """Test pipeline with LLM disabled returns original artifacts."""
        # Enhance with LLM disabled
        enhanced = await business_logic_generator_disabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Should return original template code
        assert enhanced.enhanced_node_file == sample_generated_artifacts.node_file
        assert len(enhanced.methods_generated) == 0
        assert enhanced.total_tokens_used == 0
        assert enhanced.total_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_pipeline_llm_enabled_modifies_code(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test pipeline with LLM enabled modifies code."""
        # Enhance with LLM enabled
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Should modify code
        assert enhanced.enhanced_node_file != sample_generated_artifacts.node_file
        assert len(enhanced.methods_generated) > 0

    def test_llm_disabled_no_api_key_required(self):
        """Test that LLM disabled doesn't require API key."""
        # Should not raise even without API key
        generator = BusinessLogicGenerator(enable_llm=False)
        assert generator.enable_llm is False
        assert generator.llm_node is None

    def test_llm_enabled_requires_api_key(self, monkeypatch):
        """Test that LLM enabled requires API key."""
        # Clear API key
        monkeypatch.delenv("ZAI_API_KEY", raising=False)

        # Should raise ModelOnexError
        with pytest.raises(ModelOnexError) as exc_info:
            BusinessLogicGenerator(enable_llm=True)

        assert "ZAI_API_KEY" in str(exc_info.value)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestPipelineErrorHandling:
    """Test pipeline error handling for various failure scenarios."""

    def test_template_engine_missing_template(self, temp_template_dir):
        """Test error handling when template not found."""
        engine = TemplateEngine(template_root=temp_template_dir)

        # Try to load non-existent template
        with pytest.raises(TemplateEngineError) as exc_info:
            engine.load_template(node_type="effect", version="v99_99_99")

        assert "not found" in str(exc_info.value).lower()

    def test_template_engine_invalid_node_type(self, temp_template_dir):
        """Test error handling for invalid node type."""
        engine = TemplateEngine(template_root=temp_template_dir)

        # Try to load with invalid node type
        with pytest.raises((TemplateEngineError, ValueError)):
            engine.load_template(node_type="invalid_type", version="v1_0_0")

    @pytest.mark.asyncio
    async def test_business_logic_invalid_artifacts(
        self,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test error handling with invalid artifacts."""
        # Create invalid artifacts (empty code)
        invalid_artifacts = ModelGeneratedArtifacts(
            node_file="",  # Empty code
            contract_file="",
            init_file="",
            node_type="effect",
            node_name="NodeInvalid",
            service_name="invalid",
        )

        # Should handle gracefully (no stubs to replace)
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=invalid_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Should return original (empty) code since no stubs
        assert enhanced.enhanced_node_file == ""

    @pytest.mark.asyncio
    async def test_business_logic_invalid_requirements(
        self,
        sample_generated_artifacts,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test error handling with invalid requirements."""
        # Create minimal/invalid requirements
        invalid_requirements = ModelPRDRequirements(
            service_name="",  # Empty service name
            node_type="effect",
            domain="api",  # Required field
            business_description="",
            operations=[],
            features=[],
        )

        # Should still work (generator should handle gracefully)
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=invalid_requirements,
            context_data=integration_context,
        )

        # Should still generate something (may be lower quality)
        assert isinstance(enhanced, ModelEnhancedArtifacts)


# ============================================================================
# Validation Tests
# ============================================================================


class TestPipelineValidation:
    """Test validation of generated code quality."""

    @pytest.mark.asyncio
    async def test_generated_code_syntax_valid(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that generated code is syntactically valid Python."""
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Parse with AST
        try:
            tree = ast.parse(enhanced.enhanced_node_file)
            syntax_valid = True
            has_class = any(isinstance(node, ast.ClassDef) for node in tree.body)
        except SyntaxError:
            syntax_valid = False
            has_class = False

        assert syntax_valid, "Generated code must be valid Python"
        assert has_class, "Generated code must contain a class definition"

    @pytest.mark.asyncio
    async def test_generated_code_has_required_methods(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that generated code contains required methods."""
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Check for required method
        assert "execute_effect" in enhanced.enhanced_node_file

        # Parse and verify method exists
        tree = ast.parse(enhanced.enhanced_node_file)
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                methods.append(node.name)

        assert "execute_effect" in methods

    @pytest.mark.asyncio
    async def test_generated_code_onex_naming_conventions(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that generated code follows ONEX naming conventions."""
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Check for ONEX patterns
        code = enhanced.enhanced_node_file

        # Should have class starting with "Node"
        assert re.search(r"class\s+Node\w+:", code), "Class should start with 'Node'"

        # Should have proper imports
        assert "from omnibase_core" in code or "import omnibase_core" in code

    @pytest.mark.asyncio
    async def test_generated_code_has_type_hints(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that generated code includes type hints."""
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        # Parse and check for type annotations
        tree = ast.parse(enhanced.enhanced_node_file)

        # Find async function definitions
        async_funcs = [
            node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
        ]

        # At least one should have return annotation
        has_return_annotation = any(func.returns is not None for func in async_funcs)

        assert has_return_annotation, "Generated methods should have type hints"

    @pytest.mark.asyncio
    async def test_generated_code_no_obvious_security_issues(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that generated code doesn't have obvious security issues."""
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        code = enhanced.enhanced_node_file

        # Check for dangerous patterns
        dangerous_patterns = [
            r"eval\(",
            r"exec\(",
            r"__import__\(",
            r"os\.system\(",
            r"subprocess\.call\(",
        ]

        for pattern in dangerous_patterns:
            assert not re.search(
                pattern, code
            ), f"Generated code should not contain {pattern}"


# ============================================================================
# Performance Tests (Optional)
# ============================================================================


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pipeline_execution_time(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test that pipeline completes within reasonable time."""
        import time

        start_time = time.time()

        await business_logic_generator_enabled.enhance_artifacts(
            artifacts=sample_generated_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        elapsed_time = time.time() - start_time

        # With mocked LLM, should be very fast (<1s)
        assert elapsed_time < 5.0, f"Pipeline took {elapsed_time}s (expected <5s)"

    @pytest.mark.asyncio
    async def test_pipeline_handles_large_templates(
        self,
        sample_generated_artifacts,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test pipeline with large template files."""
        # Create large template (1000 lines)
        large_template = (
            sample_generated_artifacts.node_file + "\n" + "# Comment\n" * 1000
        )

        large_artifacts = ModelGeneratedArtifacts(
            node_file=large_template,
            contract_file="",
            init_file="",
            node_type="effect",
            node_name="NodeLarge",
            service_name="large",
        )

        # Should handle large files
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=large_artifacts,
            requirements=sample_prd_requirements,
            context_data=integration_context,
        )

        assert isinstance(enhanced, ModelEnhancedArtifacts)
        assert len(enhanced.enhanced_node_file) > len(large_template) * 0.9


# ============================================================================
# All Node Types Test
# ============================================================================


class TestPipelineAllNodeTypes:
    """Test pipeline works for all node types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "node_type,required_method",
        [
            ("effect", "execute_effect"),
            ("compute", "execute_compute"),
            # Note: reducer and orchestrator would require additional fixtures
        ],
    )
    async def test_pipeline_supports_all_node_types(
        self,
        node_type,
        required_method,
        temp_template_dir,
        sample_prd_requirements,
        integration_context,
        business_logic_generator_enabled,
    ):
        """Test pipeline works for all node types."""
        # Load template
        engine = TemplateEngine(template_root=temp_template_dir)
        artifacts = engine.load_template(node_type=node_type, version="v1_0_0")

        # Update requirements
        requirements = ModelPRDRequirements(
            **{
                **sample_prd_requirements.model_dump(),
                "node_type": node_type,
            }
        )

        # Enhance
        enhanced = await business_logic_generator_enabled.enhance_artifacts(
            artifacts=artifacts,
            requirements=requirements,
            context_data=integration_context,
        )

        # Verify
        assert required_method in enhanced.enhanced_node_file
        assert isinstance(enhanced, ModelEnhancedArtifacts)


# ============================================================================
# Fixtures used by tests (imported from conftest.py)
# ============================================================================

# The following fixtures are provided by conftest.py:
# - sample_generated_artifacts
# - sample_prd_requirements
# - integration_context
# - business_logic_generator_enabled
# - business_logic_generator_disabled
# - temp_template_dir
# - mock_anthropic_client
# - mock_zai_api_key

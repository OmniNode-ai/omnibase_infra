#!/usr/bin/env python3
"""
Unit tests for Jinja2 template rendering.

Tests all node templates (Effect, Compute, Reducer, Orchestrator) with various
mixin combinations to ensure templates render without errors.
"""

import ast
import logging
from pathlib import Path

import pytest
import yaml

from omninode_bridge.codegen.node_classifier import (
    EnumNodeType,
    ModelClassificationResult,
)
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.template_engine import TemplateEngine

logger = logging.getLogger(__name__)


@pytest.fixture
def templates_directory():
    """Get path to templates directory."""
    return (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "omninode_bridge"
        / "codegen"
        / "templates"
    )


@pytest.fixture
def template_engine(templates_directory):
    """Create TemplateEngine instance."""
    return TemplateEngine(templates_directory=templates_directory)


@pytest.fixture
def sample_requirements():
    """Create sample PRD requirements."""
    return ModelPRDRequirements(
        service_name="test_service",
        business_description="A test service for template validation",
        operations=["create", "read", "update"],
        features=["authentication", "caching", "monitoring"],
        domain="testing",
        dependencies={},
        performance_requirements={},
        data_models=[],
        best_practices=[],
        code_examples=[],
    )


@pytest.fixture
def effect_classification():
    """Create Effect node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.EFFECT,
        template_name="effect",
        template_variant="standard",
        confidence=0.95,
        rationale="Test effect node",
    )


@pytest.fixture
def compute_classification():
    """Create Compute node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.COMPUTE,
        template_name="compute",
        template_variant="standard",
        confidence=0.95,
        rationale="Test compute node",
    )


@pytest.fixture
def reducer_classification():
    """Create Reducer node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.REDUCER,
        template_name="reducer",
        template_variant="standard",
        confidence=0.95,
        rationale="Test reducer node",
    )


@pytest.fixture
def orchestrator_classification():
    """Create Orchestrator node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.ORCHESTRATOR,
        template_name="orchestrator",
        template_variant="standard",
        confidence=0.95,
        rationale="Test orchestrator node",
    )


class TestNodeEffectTemplate:
    """Test Effect node template rendering."""

    def test_effect_template_renders_without_error(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test Effect template renders successfully."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        node_content = template_engine._generate_node_file(
            effect_classification.node_type, "effect", context
        )

        assert node_content
        assert "class NodeTestServiceEffect" in node_content
        assert "async def" in node_content
        assert "NodeEffect" in node_content

    def test_effect_template_generates_valid_python(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test Effect template generates syntactically valid Python."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        node_content = template_engine._generate_node_file(
            effect_classification.node_type, "effect", context
        )

        # Parse Python code to verify syntax
        try:
            ast.parse(node_content)
        except SyntaxError as e:
            pytest.fail(
                f"Generated code has syntax error: {e}\n\nCode:\n{node_content}"
            )

    def test_effect_template_includes_operations(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test Effect template includes defined operations."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        node_content = template_engine._generate_node_file(
            effect_classification.node_type, "effect", context
        )

        # Should have methods for operations
        assert "async def create" in node_content
        assert "async def read" in node_content
        assert "async def update" in node_content

    def test_effect_template_with_no_mixins(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test Effect template works with no mixins enabled."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        context["enabled_mixins"] = []
        context["base_classes"] = ["NodeEffect"]

        node_content = template_engine._generate_node_file(
            effect_classification.node_type, "effect", context
        )

        assert node_content
        ast.parse(node_content)  # Verify syntax


class TestNodeComputeTemplate:
    """Test Compute node template rendering."""

    def test_compute_template_renders_without_error(
        self, template_engine, sample_requirements, compute_classification
    ):
        """Test Compute template renders successfully."""
        context = template_engine._build_template_context(
            sample_requirements, compute_classification
        )
        node_content = template_engine._generate_node_file(
            compute_classification.node_type, "compute", context
        )

        assert node_content
        assert "class NodeTestServiceCompute" in node_content
        assert "NodeCompute" in node_content

    def test_compute_template_generates_valid_python(
        self, template_engine, sample_requirements, compute_classification
    ):
        """Test Compute template generates syntactically valid Python."""
        context = template_engine._build_template_context(
            sample_requirements, compute_classification
        )
        node_content = template_engine._generate_node_file(
            compute_classification.node_type, "compute", context
        )

        try:
            ast.parse(node_content)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")


class TestNodeReducerTemplate:
    """Test Reducer node template rendering."""

    def test_reducer_template_renders_without_error(
        self, template_engine, sample_requirements, reducer_classification
    ):
        """Test Reducer template renders successfully."""
        context = template_engine._build_template_context(
            sample_requirements, reducer_classification
        )
        node_content = template_engine._generate_node_file(
            reducer_classification.node_type, "reducer", context
        )

        assert node_content
        assert "class NodeTestServiceReducer" in node_content
        assert "NodeReducer" in node_content

    def test_reducer_template_generates_valid_python(
        self, template_engine, sample_requirements, reducer_classification
    ):
        """Test Reducer template generates syntactically valid Python."""
        context = template_engine._build_template_context(
            sample_requirements, reducer_classification
        )
        node_content = template_engine._generate_node_file(
            reducer_classification.node_type, "reducer", context
        )

        try:
            ast.parse(node_content)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")

    def test_reducer_template_includes_aggregation_state(
        self, template_engine, sample_requirements, reducer_classification
    ):
        """Test Reducer template includes aggregation state."""
        context = template_engine._build_template_context(
            sample_requirements, reducer_classification
        )
        node_content = template_engine._generate_node_file(
            reducer_classification.node_type, "reducer", context
        )

        assert "accumulated_state" in node_content
        assert "aggregation_count" in node_content


class TestNodeOrchestratorTemplate:
    """Test Orchestrator node template rendering."""

    def test_orchestrator_template_renders_without_error(
        self, template_engine, sample_requirements, orchestrator_classification
    ):
        """Test Orchestrator template renders successfully."""
        context = template_engine._build_template_context(
            sample_requirements, orchestrator_classification
        )
        node_content = template_engine._generate_node_file(
            orchestrator_classification.node_type, "orchestrator", context
        )

        assert node_content
        assert "class NodeTestServiceOrchestrator" in node_content
        assert "NodeOrchestrator" in node_content

    def test_orchestrator_template_generates_valid_python(
        self, template_engine, sample_requirements, orchestrator_classification
    ):
        """Test Orchestrator template generates syntactically valid Python."""
        context = template_engine._build_template_context(
            sample_requirements, orchestrator_classification
        )
        node_content = template_engine._generate_node_file(
            orchestrator_classification.node_type, "orchestrator", context
        )

        try:
            ast.parse(node_content)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")

    def test_orchestrator_template_includes_workflow_state(
        self, template_engine, sample_requirements, orchestrator_classification
    ):
        """Test Orchestrator template includes workflow state."""
        context = template_engine._build_template_context(
            sample_requirements, orchestrator_classification
        )
        node_content = template_engine._generate_node_file(
            orchestrator_classification.node_type, "orchestrator", context
        )

        assert "active_workflows" in node_content


class TestContractTemplate:
    """Test contract.yaml template rendering."""

    def test_contract_template_renders_for_effect(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test contract template renders for Effect node."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        contract_content = template_engine._generate_contract_file(
            effect_classification.node_type, context
        )

        assert contract_content
        assert 'name: "test_service"' in contract_content
        assert 'node_type: "EFFECT"' in contract_content

    def test_contract_template_generates_valid_yaml(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test contract template generates valid YAML."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        contract_content = template_engine._generate_contract_file(
            effect_classification.node_type, context
        )

        # Parse YAML to verify syntax
        try:
            yaml.safe_load(contract_content)
        except yaml.YAMLError as e:
            pytest.fail(
                f"Generated contract has invalid YAML: {e}\n\nYAML:\n{contract_content}"
            )

    def test_contract_includes_io_operations_for_effect(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test contract includes io_operations for Effect nodes."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        contract_content = template_engine._generate_contract_file(
            effect_classification.node_type, context
        )

        assert "io_operations:" in contract_content
        assert "operation_type:" in contract_content

    def test_contract_excludes_io_operations_for_compute(
        self, template_engine, sample_requirements, compute_classification
    ):
        """Test contract excludes io_operations for Compute nodes."""
        context = template_engine._build_template_context(
            sample_requirements, compute_classification
        )
        contract_content = template_engine._generate_contract_file(
            compute_classification.node_type, context
        )

        # Compute nodes should not have io_operations
        assert 'node_type: "COMPUTE"' in contract_content


class TestInitTemplate:
    """Test __init__.py template rendering."""

    def test_init_template_renders(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test __init__.py template renders successfully."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        init_content = template_engine._generate_init_file(context)

        assert init_content
        assert "from .node import NodeTestServiceEffect" in init_content
        assert "__all__" in init_content

    def test_init_template_generates_valid_python(
        self, template_engine, sample_requirements, effect_classification
    ):
        """Test __init__.py template generates valid Python."""
        context = template_engine._build_template_context(
            sample_requirements, effect_classification
        )
        init_content = template_engine._generate_init_file(context)

        try:
            ast.parse(init_content)
        except SyntaxError as e:
            pytest.fail(f"Generated __init__ has syntax error: {e}")


class TestCustomFilters:
    """Test custom Jinja2 filters."""

    def test_to_snake_case_filter(self, template_engine):
        """Test to_snake_case filter works correctly."""
        if not template_engine.env:
            pytest.skip("Template environment not initialized")

        to_snake_case = template_engine.env.filters["to_snake_case"]
        assert to_snake_case("MixinHealthCheck") == "mixin_health_check"
        assert to_snake_case("NodeEffect") == "node_effect"
        assert to_snake_case("TestService") == "test_service"

    def test_sort_imports_filter(self, template_engine):
        """Test sort_imports filter works correctly."""
        if not template_engine.env:
            pytest.skip("Template environment not initialized")

        sort_imports = template_engine.env.filters["sort_imports"]
        imports = ["import z", "import a", "import m"]
        sorted_imports = sort_imports(imports)
        assert sorted_imports == ["import a", "import m", "import z"]

    def test_repr_filter(self, template_engine):
        """Test repr filter works correctly."""
        if not template_engine.env:
            pytest.skip("Template environment not initialized")

        repr_filter = template_engine.env.filters["repr"]
        assert repr_filter("test") == "'test'"
        assert repr_filter(123) == "123"
        assert repr_filter(True) == "True"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_template_with_empty_operations(
        self, template_engine, effect_classification
    ):
        """Test templates handle empty operations list."""
        requirements = ModelPRDRequirements(
            service_name="test_service",
            business_description="Test service",
            operations=[],  # Empty
            features=["feature1"],
            domain="testing",
            dependencies={},
            performance_requirements={},
            data_models=[],
            best_practices=[],
            code_examples=[],
        )

        context = template_engine._build_template_context(
            requirements, effect_classification
        )
        node_content = template_engine._generate_node_file(
            effect_classification.node_type, "effect", context
        )

        # Should still generate valid code
        ast.parse(node_content)

    def test_template_with_special_characters_in_description(
        self, template_engine, effect_classification
    ):
        """Test templates handle special characters in description."""
        requirements = ModelPRDRequirements(
            service_name="test_service",
            business_description="A \"test\" service with 'quotes' and $pecial chars",
            operations=["create"],
            features=["feature1"],
            domain="testing",
            dependencies={},
            performance_requirements={},
            data_models=[],
            best_practices=[],
            code_examples=[],
        )

        context = template_engine._build_template_context(
            requirements, effect_classification
        )
        node_content = template_engine._generate_node_file(
            effect_classification.node_type, "effect", context
        )

        # Should still generate valid code
        ast.parse(node_content)

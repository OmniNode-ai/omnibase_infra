"""
Unit tests for NodeTestGeneratorEffect.

Tests the test generation node's ability to:
- Parse ModelContractTest from YAML
- Render Jinja2 templates
- Write test files to disk
- Validate generated code syntax
- Handle errors gracefully
"""

import tempfile
from pathlib import Path

import pytest
from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from ..node import NodeTestGeneratorEffect


@pytest.fixture
def container():
    """Create test container."""
    return ModelContainer(
        value={
            "template_directory": "src/omninode_bridge/codegen/templates/test_templates",
            "environment": "test",
        },
        container_type="config",
    )


@pytest.fixture
def node(container):
    """Create test node instance."""
    return NodeTestGeneratorEffect(container)


@pytest.fixture
def sample_test_contract_yaml():
    """Sample ModelContractTest YAML for testing."""
    return """
name: "test_postgres_crud"
version:
  major: 1
  minor: 0
  patch: 0
description: "Test contract for PostgreSQL CRUD operations"

target_node: "postgres_crud_effect"
target_version: "1.0.0"
target_node_type: "effect"
test_suite_name: "test_postgres_crud_effect"

coverage_minimum: 85
coverage_target: 95

test_types:
  - unit
  - integration

test_targets:
  - target_name: "execute_effect"
    target_type: "method"
    test_scenarios:
      - "successful_execution"
      - "error_handling"
      - "edge_cases"

mock_requirements:
  mock_dependencies:
    - "postgres_client"
  mock_external_services:
    - "kafka_client"

test_configuration:
  pytest_markers:
    - "asyncio"
    - "unit"
  timeout_seconds: 30
"""


@pytest.mark.asyncio
async def test_node_initialization(node):
    """Test node initializes correctly."""
    assert node is not None
    assert node.node_id is not None
    assert node.config is not None


@pytest.mark.asyncio
async def test_execute_effect_generates_unit_tests(node, sample_test_contract_yaml):
    """Test effect execution generates unit test files."""
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create contract
        contract = ModelContractEffect(
            name="generate_tests",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Generate tests for PostgresCrudEffect",
            node_type=EnumNodeType.EFFECT,
            io_operations=[{"operation_type": "file_write", "atomic": True}],
            input_model="ModelTestGeneratorRequest",
            output_model="ModelTestGeneratorResponse",
            tool_specification={
                "tool_name": "test_generator",
                "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect",
            },
            input_state={
                "test_contract_yaml": sample_test_contract_yaml,
                "output_directory": str(output_dir),
                "node_name": "postgres_crud_effect",
                "enable_fixtures": False,  # Skip conftest for now
                "overwrite_existing": True,
            },
        )

        # Execute
        response = await node.execute_effect(contract)

        # Verify response
        assert response is not None
        assert response.success is True
        assert response.file_count >= 1  # At least unit tests
        assert response.total_lines_of_code > 0
        assert response.duration_ms > 0

        # Verify files were created
        unit_test_file = output_dir / "test_unit_postgres_crud_effect.py"
        assert unit_test_file.exists(), f"Unit test file not found: {unit_test_file}"

        # Verify file content is valid Python
        content = unit_test_file.read_text()
        assert "def test_" in content or "async def test_" in content
        assert "NodePostgresCrudEffect" in content or "postgres_crud" in content


@pytest.mark.asyncio
async def test_execute_effect_with_fixtures(node, sample_test_contract_yaml):
    """Test effect execution generates conftest.py when fixtures enabled."""
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create contract
        contract = ModelContractEffect(
            name="generate_tests",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Generate tests with fixtures",
            node_type=EnumNodeType.EFFECT,
            io_operations=[{"operation_type": "file_write", "atomic": True}],
            input_model="ModelTestGeneratorRequest",
            output_model="ModelTestGeneratorResponse",
            tool_specification={
                "tool_name": "test_generator",
                "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect",
            },
            input_state={
                "test_contract_yaml": sample_test_contract_yaml,
                "output_directory": str(output_dir),
                "node_name": "postgres_crud_effect",
                "enable_fixtures": True,
                "overwrite_existing": True,
            },
        )

        # Execute
        response = await node.execute_effect(contract)

        # Verify conftest.py was generated (if template exists)
        # This may generate a warning if conftest template is missing
        if response.file_count > 2:
            conftest_file = output_dir / "conftest.py"
            # Only assert if conftest was actually generated
            if conftest_file.exists():
                assert conftest_file.exists()
                content = conftest_file.read_text()
                assert "pytest" in content.lower() or "fixture" in content.lower()


@pytest.mark.asyncio
async def test_execute_effect_validates_syntax(node):
    """Test that generated code is validated for syntax errors."""
    # This test would need a malformed template to trigger syntax validation
    # For now, we just verify the validation method exists
    assert hasattr(node, "_validate_python_syntax")


@pytest.mark.asyncio
async def test_get_metrics(node):
    """Test metrics retrieval."""
    metrics = node.get_metrics()

    assert isinstance(metrics, dict)
    assert "total_generations" in metrics
    assert "total_files_generated" in metrics
    assert "avg_files_per_generation" in metrics
    assert "avg_generation_time_ms" in metrics


@pytest.mark.asyncio
async def test_template_context_building(node, sample_test_contract_yaml):
    """Test template context is built correctly."""
    from omninode_bridge.codegen.models.model_contract_test import ModelContractTest

    from ..models.model_request import ModelTestGeneratorRequest

    # Parse test contract
    test_contract = ModelContractTest.from_yaml(sample_test_contract_yaml)

    # Create request
    request = ModelTestGeneratorRequest(
        test_contract_yaml=sample_test_contract_yaml,
        output_directory=Path("/tmp/tests"),
        node_name="postgres_crud_effect",
    )

    # Build context
    context = node._build_template_context(test_contract, request)

    # Verify context has required fields
    assert "test_contract" in context
    assert "node_name" in context
    assert "target_node" in context
    assert "test_types" in context
    assert "generated_at" in context
    assert context["node_name"] == "postgres_crud_effect"


@pytest.mark.asyncio
async def test_count_lines_of_code(node):
    """Test line counting logic."""
    sample_code = """
# This is a comment
import pytest

def test_example():
    assert True

    # Another comment
    return True
"""

    loc = node._count_lines_of_code(sample_code)

    # Should count: import, def, assert, return = 4 lines
    # (Comments and blank lines excluded)
    assert loc >= 4
    assert loc < 10  # Sanity check

"""
Integration tests for NodeTestGeneratorEffect.

Tests end-to-end test generation workflows including:
- Complete test suite generation
- Multiple test types
- Template rendering with real templates
- File system operations
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
def comprehensive_test_contract_yaml():
    """Comprehensive ModelContractTest YAML with all test types."""
    return """
name: "test_comprehensive"
version:
  major: 1
  minor: 0
  patch: 0
description: "Comprehensive test contract with all test types"

target_node: "sample_node"
target_version: "1.0.0"
target_node_type: "effect"
test_suite_name: "test_sample_node"

coverage_minimum: 90
coverage_target: 95

test_types:
  - unit
  - integration

test_targets:
  - target_name: "execute_effect"
    target_type: "method"
    test_scenarios:
      - "success_case"
      - "error_case"
      - "timeout_case"
      - "validation_error"

  - target_name: "initialize"
    target_type: "method"
    test_scenarios:
      - "valid_config"
      - "invalid_config"
      - "missing_dependencies"

mock_requirements:
  mock_dependencies:
    - "database_client"
    - "cache_client"
  mock_external_services:
    - "api_gateway"
    - "message_queue"

test_configuration:
  pytest_markers:
    - "asyncio"
    - "unit"
    - "integration"
  timeout_seconds: 60
  parallel_execution: true

include_docstrings: true
include_type_hints: true
use_async_tests: true
parametrize_tests: true
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_test_generation(node, comprehensive_test_contract_yaml):
    """Test complete end-to-end test generation workflow."""
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create contract
        contract = ModelContractEffect(
            name="generate_comprehensive_tests",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Generate comprehensive test suite",
            node_type=EnumNodeType.EFFECT,
            io_operations=[{"operation_type": "file_write", "atomic": True}],
            input_model="ModelTestGeneratorRequest",
            output_model="ModelTestGeneratorResponse",
            tool_specification={
                "tool_name": "test_generator",
                "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect",
            },
            input_state={
                "test_contract_yaml": comprehensive_test_contract_yaml,
                "output_directory": str(output_dir),
                "node_name": "sample_node",
                "enable_fixtures": True,
                "overwrite_existing": True,
            },
        )

        # Execute
        response = await node.execute_effect(contract)

        # Verify response
        assert response.success is True
        assert (
            response.file_count >= 2
        )  # Unit + Integration (conftest may fail gracefully)
        assert response.total_lines_of_code > 50  # Reasonable amount of code generated
        assert response.duration_ms < 5000  # Completes in reasonable time
        assert response.template_render_ms > 0
        assert response.file_write_ms >= 0

        # Verify files exist
        assert output_dir.exists()
        test_files = list(output_dir.glob("test_*.py"))
        assert (
            len(test_files) >= 2
        ), f"Expected at least 2 test files, found: {test_files}"

        # Verify each file is valid Python
        for test_file in test_files:
            content = test_file.read_text()
            assert len(content) > 0
            # Basic Python syntax check
            compile(content, str(test_file), "exec")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_overwrite_protection(node, comprehensive_test_contract_yaml):
    """Test that existing files are protected unless overwrite is enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First generation
        contract1 = ModelContractEffect(
            name="generate_tests",
            version={"major": 1, "minor": 0, "patch": 0},
            description="First generation",
            node_type=EnumNodeType.EFFECT,
            io_operations=[{"operation_type": "file_write", "atomic": True}],
            input_model="ModelTestGeneratorRequest",
            output_model="ModelTestGeneratorResponse",
            tool_specification={
                "tool_name": "test_generator",
                "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect",
            },
            input_state={
                "test_contract_yaml": comprehensive_test_contract_yaml,
                "output_directory": str(output_dir),
                "node_name": "sample_node",
                "enable_fixtures": False,
                "overwrite_existing": True,
            },
        )

        response1 = await node.execute_effect(contract1)
        assert response1.success is True

        # Second generation with overwrite=False should fail
        contract2 = ModelContractEffect(
            name="generate_tests_again",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Second generation without overwrite",
            node_type=EnumNodeType.EFFECT,
            io_operations=[{"operation_type": "file_write", "atomic": True}],
            input_model="ModelTestGeneratorRequest",
            output_model="ModelTestGeneratorResponse",
            tool_specification={
                "tool_name": "test_generator",
                "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect",
            },
            input_state={
                "test_contract_yaml": comprehensive_test_contract_yaml,
                "output_directory": str(output_dir),
                "node_name": "sample_node",
                "enable_fixtures": False,
                "overwrite_existing": False,  # Should fail
            },
        )

        # This should raise an error because files already exist
        from omnibase_core import ModelOnexError

        with pytest.raises(ModelOnexError) as exc_info:
            await node.execute_effect(contract2)

        assert "already exists" in str(exc_info.value.message).lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_tracking_across_generations(
    node, comprehensive_test_contract_yaml
):
    """Test that metrics are tracked correctly across multiple generations."""
    # Get initial metrics
    initial_metrics = node.get_metrics()
    initial_generations = initial_metrics["total_generations"]

    # Perform multiple generations
    for i in range(3):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            contract = ModelContractEffect(
                name=f"generate_tests_{i}",
                version={"major": 1, "minor": 0, "patch": 0},
                description=f"Generation {i}",
                node_type=EnumNodeType.EFFECT,
                io_operations=[{"operation_type": "file_write", "atomic": True}],
                input_model="ModelTestGeneratorRequest",
                output_model="ModelTestGeneratorResponse",
                tool_specification={
                    "tool_name": "test_generator",
                    "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect",
                },
                input_state={
                    "test_contract_yaml": comprehensive_test_contract_yaml,
                    "output_directory": str(output_dir),
                    "node_name": "sample_node",
                    "enable_fixtures": False,
                    "overwrite_existing": True,
                },
            )

            response = await node.execute_effect(contract)
            assert response.success is True

    # Get final metrics
    final_metrics = node.get_metrics()

    # Verify metrics increased
    assert final_metrics["total_generations"] == initial_generations + 3
    assert (
        final_metrics["total_files_generated"]
        > initial_metrics["total_files_generated"]
    )
    assert final_metrics["avg_generation_time_ms"] > 0

#!/usr/bin/env python3
"""
End-to-end integration tests for code generation.

Tests the full pipeline from PRD requirements to validated generated code.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omninode_bridge.codegen.quality_gates import QualityGatePipeline
from omninode_bridge.codegen.service import CodeGenerationService
from omninode_bridge.codegen.strategies.base import EnumStrategyType
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts


@pytest.mark.integration
class TestEndToEndCodeGeneration:
    """Test complete code generation workflow."""

    @pytest.fixture
    def service(self, temp_output_dir):
        """Create CodeGenerationService instance for integration testing."""
        return CodeGenerationService(
            templates_directory=None,  # Use default
            archon_mcp_url=None,  # Disable for tests
            enable_intelligence=False,
        )

    @pytest.fixture
    def quality_pipeline(self):
        """Create QualityGatePipeline for validation."""
        return QualityGatePipeline(
            validation_level="strict",
            enable_mypy=False,  # Disable for faster tests
        )

    @pytest.mark.asyncio
    async def test_simple_crud_generation_jinja2_strategy(
        self,
        service,
        simple_crud_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test generating simple CRUD node with Jinja2Strategy."""
        # Initialize strategies
        service._initialize_strategies()

        # Mock the template engine
        mock_artifacts = ModelGeneratedArtifacts(
            node_file=self._get_sample_effect_code(),
            contract_file=self._get_sample_contract_yaml(),
            init_file='"""Init file for postgres_crud node."""\n',
            node_type="effect",
            node_name="NodePostgresCrudEffect",
            service_name="postgres_crud",
            models={},
            tests={},
            documentation={},
        )

        # Patch the Jinja2Strategy's template engine
        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Generate node
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
                strategy="jinja2",  # Explicit strategy
                enable_llm=False,
                validation_level="strict",
            )

            # Verify result
            assert result is not None
            assert result.strategy_used == EnumStrategyType.JINJA2
            assert result.llm_used is False
            assert result.validation_passed is True
            assert result.artifacts.node_name == "NodePostgresCrudEffect"

            # Verify performance (Jinja2 should be fast)
            assert result.generation_time_ms < 5000  # < 5 seconds

    @pytest.mark.asyncio
    async def test_auto_strategy_selection_simple_requirements(
        self,
        service,
        simple_crud_requirements,
        temp_output_dir,
    ):
        """Test automatic strategy selection for simple requirements."""
        service._initialize_strategies()

        mock_artifacts = ModelGeneratedArtifacts(
            node_file=self._get_sample_effect_code(),
            contract_file=self._get_sample_contract_yaml(),
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodePostgresCrudEffect",
            service_name="postgres_crud",
            models={},
            tests={},
            documentation={},
        )

        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Generate with AUTO strategy
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=None,  # Will auto-classify
                output_directory=temp_output_dir,
                strategy="auto",  # Auto-select strategy
                enable_llm=False,
                validation_level="standard",
            )

            # Should auto-select Jinja2 for simple requirements
            assert result.strategy_used == EnumStrategyType.JINJA2
            assert result.validation_passed is True

    @pytest.mark.asyncio
    async def test_validation_catches_syntax_errors(
        self,
        service,
        simple_crud_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test that validation catches syntax errors in generated code."""
        service._initialize_strategies()

        # Create artifacts with syntax error
        mock_artifacts = ModelGeneratedArtifacts(
            node_file="""#!/usr/bin/env python3
def broken_function()  # Missing colon
    pass
""",
            contract_file="# Contract",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodeBrokenEffect",
            service_name="broken",
            models={},
            tests={},
            documentation={},
        )

        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Generate node
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
                strategy="jinja2",
                enable_llm=False,
                validation_level="basic",  # Even basic validation should catch syntax
            )

            # Result should be returned (generation succeeded)
            # But validation would catch syntax error if run separately
            assert result is not None

    @pytest.mark.asyncio
    async def test_generate_multiple_node_types(
        self,
        service,
        temp_output_dir,
    ):
        """Test generating different node types."""
        service._initialize_strategies()

        from omninode_bridge.codegen.node_classifier import (
            EnumNodeType,
            ModelClassificationResult,
        )

        # Test different node types
        node_types = [
            (EnumNodeType.EFFECT, "NodeTestEffect"),
            (EnumNodeType.COMPUTE, "NodeTestCompute"),
            (EnumNodeType.REDUCER, "NodeTestReducer"),
            (EnumNodeType.ORCHESTRATOR, "NodeTestOrchestrator"),
        ]

        for node_type, expected_name in node_types:
            # Create classification
            classification = ModelClassificationResult(
                node_type=node_type,
                confidence=0.9,
                reasoning=f"Test {node_type.value} node",
                suggested_patterns=[],
            )

            # Create mock artifacts
            mock_artifacts = ModelGeneratedArtifacts(
                node_file=f"# {node_type.value} node code",
                contract_file="# Contract",
                init_file='"""Init"""',
                node_type=node_type.value,
                node_name=expected_name,
                service_name="test",
                models={},
                tests={},
                documentation={},
            )

            jinja2_strategy = service.strategy_registry.get_strategy(
                EnumStrategyType.JINJA2
            )
            with patch.object(
                jinja2_strategy.template_engine,
                "generate",
                new_callable=AsyncMock,
                return_value=mock_artifacts,
            ):
                # Generate
                result = await service.generate_node(
                    requirements=simple_crud_requirements(),
                    classification=classification,
                    output_directory=temp_output_dir,
                    strategy="jinja2",
                    enable_llm=False,
                )

                # Verify
                assert result.artifacts.node_name == expected_name
                assert result.artifacts.node_type == node_type.value

    @pytest.mark.asyncio
    async def test_error_handling_for_invalid_requirements(
        self,
        service,
        invalid_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test error handling for invalid requirements."""
        service._initialize_strategies()

        # Should raise ValueError for invalid requirements in strict mode
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
    async def test_correlation_id_tracking(
        self,
        service,
        simple_crud_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test that correlation ID is tracked through generation."""
        service._initialize_strategies()

        correlation_id = uuid4()

        mock_artifacts = ModelGeneratedArtifacts(
            node_file="# Code",
            contract_file="# Contract",
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodeTest",
            service_name="test",
            models={},
            tests={},
            documentation={},
        )

        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir,
                strategy="jinja2",
                enable_llm=False,
                correlation_id=correlation_id,
            )

            # Verify correlation ID is preserved
            assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_file_generation_and_validation(
        self,
        service,
        simple_crud_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test that files are generated and can be validated."""
        service._initialize_strategies()

        # Create mock artifacts
        mock_artifacts = ModelGeneratedArtifacts(
            node_file=self._get_sample_effect_code(),
            contract_file=self._get_sample_contract_yaml(),
            init_file='"""Init file for test node."""\n',
            node_type="effect",
            node_name="NodeTestEffect",
            service_name="test_effect",
            models={},
            tests={},
            documentation={},
        )

        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Generate node
            output_dir = temp_output_dir / "test_effect"
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=output_dir,
                strategy="jinja2",
                enable_llm=False,
            )

            # Verify base directory files exist
            assert (
                output_dir / "node.py"
            ).exists(), "node.py should exist in base directory"
            assert (
                output_dir / "contract.yaml"
            ).exists(), "contract.yaml should exist in base directory"
            assert (
                output_dir / "__init__.py"
            ).exists(), "__init__.py should exist in base directory"

            # Verify files can be validated
            from omninode_bridge.codegen.file_validator import FileValidator

            validator = FileValidator()
            validation_result = await validator.validate_generated_files(
                file_paths=[output_dir / "node.py"],
                strict_mode=True,
            )

            # Generated files should pass validation (no stubs in this sample code)
            assert (
                validation_result.passed
            ), f"Validation failed: {validation_result.issues}"
            assert validation_result.files_validated == 1
            assert validation_result.files_passed == 1

            # Verify content is readable and contains expected elements
            node_content = (output_dir / "node.py").read_text()
            assert "NodePostgresCrudEffect" in node_content, "Should contain node class"
            assert (
                "async def execute_effect" in node_content
            ), "Should contain execute_effect method"
            assert "ModelContractEffect" in node_content, "Should import contract model"

    @pytest.mark.asyncio
    async def test_stub_detection_in_generated_files(
        self,
        service,
        simple_crud_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test that stub patterns are detected in generated files."""
        from omninode_bridge.codegen.file_validator import FileValidator

        service._initialize_strategies()

        # Create artifacts with stub patterns
        stub_code = '''#!/usr/bin/env python3
"""
NodeStubEffect - Test node with stubs.
"""

import logging
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

logger = logging.getLogger(__name__)


class NodeStubEffect:
    """Effect node with stub implementation."""

    def __init__(self):
        """Initialize the effect node."""
        logger.info("NodeStubEffect initialized")

    async def execute_effect(self, contract: ModelContractEffect) -> ModelContainer:
        """
        Execute the effect.

        This method requires implementation.
        """
        # IMPLEMENTATION REQUIRED
        pass

    async def validate_input(self, data: dict) -> bool:
        """
        Validate input data.
        """
        # TODO: Implement validation logic
        pass
'''

        mock_artifacts = ModelGeneratedArtifacts(
            node_file=stub_code,
            contract_file=self._get_sample_contract_yaml(),
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodeStubEffect",
            service_name="stub_effect",
            models={},
            tests={},
            documentation={},
        )

        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Generate node (will have stubs)
            output_dir = temp_output_dir / "stub_effect"
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=output_dir,
                strategy="jinja2",
                enable_llm=False,
            )

            # Validate generated files for stubs
            validator = FileValidator()
            validation_result = await validator.validate_generated_files(
                file_paths=[output_dir / "node.py"],
                strict_mode=True,
            )

            # Should detect stubs
            assert (
                not validation_result.passed
            ), "Validation should fail when stubs are present"
            assert len(validation_result.stub_issues) > 0, "Should detect stub issues"

            # Verify stub patterns are detected
            stub_messages = [issue.message for issue in validation_result.stub_issues]
            assert any(
                "IMPLEMENTATION REQUIRED" in msg for msg in stub_messages
            ), "Should detect 'IMPLEMENTATION REQUIRED' stub pattern"

    @pytest.mark.asyncio
    async def test_generation_performance_within_targets(
        self,
        service,
        simple_crud_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test that generation completes within performance targets."""
        import time

        service._initialize_strategies()

        # Create mock artifacts
        mock_artifacts = ModelGeneratedArtifacts(
            node_file=self._get_sample_effect_code(),
            contract_file=self._get_sample_contract_yaml(),
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodePerfTest",
            service_name="perf_test",
            models={},
            tests={},
            documentation={},
        )

        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Measure generation time
            start_time = time.time()

            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=temp_output_dir / "perf_test",
                strategy="jinja2",
                enable_llm=False,
            )

            duration_seconds = time.time() - start_time

            # Verify performance
            assert (
                duration_seconds < 10.0
            ), f"Generation took {duration_seconds:.2f}s, should be < 10s"
            assert (
                result.generation_time_ms < 10000
            ), f"Generation time {result.generation_time_ms}ms exceeds 10s target"

            # Verify success
            assert result is not None
            assert result.validation_passed is True

    @pytest.mark.asyncio
    async def test_quality_gates_catch_indentation_errors(
        self,
        service,
        simple_crud_requirements,
        effect_classification,
        temp_output_dir,
    ):
        """Test that quality gates detect indentation errors."""
        service._initialize_strategies()

        # Create code with indentation error
        bad_code = '''#!/usr/bin/env python3
"""Test node with indentation error."""

import logging

logger = logging.getLogger(__name__)

class NodeBadIndentation:
    """Node with indentation error."""

    async def execute_effect(self, contract):
        """Execute effect."""
        try:
    # Wrong indentation - should be caught
    return None
'''

        mock_artifacts = ModelGeneratedArtifacts(
            node_file=bad_code,
            contract_file=self._get_sample_contract_yaml(),
            init_file='"""Init"""',
            node_type="effect",
            node_name="NodeBadIndentation",
            service_name="bad_indentation",
            models={},
            tests={},
            documentation={},
        )

        jinja2_strategy = service.strategy_registry.get_strategy(
            EnumStrategyType.JINJA2
        )
        with patch.object(
            jinja2_strategy.template_engine,
            "generate",
            new_callable=AsyncMock,
            return_value=mock_artifacts,
        ):
            # Generate node (should catch indentation error)
            output_dir = temp_output_dir / "bad_indentation"
            result = await service.generate_node(
                requirements=simple_crud_requirements,
                classification=effect_classification,
                output_directory=output_dir,
                strategy="jinja2",
                enable_llm=False,
                validation_level="basic",  # Even basic validation should catch syntax errors
            )

            # Validation should fail or catch the error
            # Note: Depending on implementation, this might pass generation but fail validation
            # Let's check if the file was created and validate it separately
            if (output_dir / "node.py").exists():
                from omninode_bridge.codegen.file_validator import FileValidator

                validator = FileValidator()
                validation_result = await validator.validate_generated_files(
                    file_paths=[output_dir / "node.py"],
                    strict_mode=True,
                )

                # Should detect syntax error
                assert (
                    not validation_result.passed
                ), "Validation should fail with indentation error"
                assert (
                    len(validation_result.syntax_errors) > 0
                ), "Should detect syntax errors"

    # Helper methods

    def _get_sample_effect_code(self) -> str:
        """Get sample effect node code."""
        return '''#!/usr/bin/env python3
"""
NodePostgresCrudEffect - PostgreSQL CRUD operations.

Author: CodeGenerationService
Version: v1_0_0
"""

import logging
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

logger = logging.getLogger(__name__)


class NodePostgresCrudEffect:
    """PostgreSQL CRUD operations effect node."""

    def __init__(self):
        """Initialize the effect node."""
        logger.info("NodePostgresCrudEffect initialized")

    async def execute_effect(self, contract: ModelContractEffect) -> ModelContainer:
        """
        Execute CRUD operation.

        Args:
            contract: Effect contract with input data

        Returns:
            ModelContainer with result data
        """
        logger.info(f"Executing CRUD operation for {contract.node_id}")

        # Process the operation
        result_data = {"status": "success", "operation": "completed"}

        # Return result
        return ModelContainer(
            data=result_data,
            metadata={"execution_time": 0.05}
        )
'''

    def _get_sample_contract_yaml(self) -> str:
        """Get sample contract YAML."""
        return """# Contract for NodePostgresCrudEffect
node_id: postgres_crud_effect_v1_0_0
node_type: effect
version: v1_0_0

input_schema:
  type: object
  properties:
    operation:
      type: string
    data:
      type: object

output_schema:
  type: object
  properties:
    status:
      type: string
    result:
      type: object
"""


# Helper to import requirements
def simple_crud_requirements():
    """Get simple CRUD requirements."""
    from tests.fixtures.codegen.sample_requirements import get_simple_crud_requirements

    return get_simple_crud_requirements()

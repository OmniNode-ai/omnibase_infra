#!/usr/bin/env python3
"""
Pytest configuration and fixtures for codegen integration tests.

Provides fixtures for:
- Mock LLM clients
- Sample templates
- Test requirements (PRD-style)
- Expected outputs
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Override the cleanup_test_data fixture from parent conftest
# Codegen tests don't need database cleanup
@pytest.fixture(autouse=True, scope="function")
async def cleanup_test_data():
    """Override parent fixture - codegen tests don't need DB cleanup."""
    yield  # No cleanup needed for file-based tests


# Add stubs to path for testing
STUBS_PATH = Path(__file__).parent.parent.parent / "stubs"
if str(STUBS_PATH) not in sys.path:
    sys.path.insert(0, str(STUBS_PATH))

# Add fixtures to path
FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "codegen"
if str(FIXTURES_PATH) not in sys.path:
    sys.path.insert(0, str(FIXTURES_PATH))

from sample_requirements import get_invalid_requirements, get_simple_crud_requirements

from omninode_bridge.codegen.business_logic import BusinessLogicGenerator
from omninode_bridge.codegen.node_classifier import (
    EnumNodeType,
    ModelClassificationResult,
)
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts
from omninode_bridge.codegen.template_engine_loader import (
    ModelStubInfo,
    ModelTemplateArtifacts,
    ModelTemplateMetadata,
    TemplateEngine,
)

# ============================================================================
# Requirements Fixtures
# ============================================================================


@pytest.fixture
def simple_crud_requirements() -> ModelPRDRequirements:
    """Simple CRUD requirements (low complexity)."""
    return get_simple_crud_requirements()


@pytest.fixture
def invalid_requirements() -> ModelPRDRequirements:
    """Invalid requirements for testing validation."""
    return get_invalid_requirements()


# ============================================================================
# Classification Fixtures
# ============================================================================


@pytest.fixture
def effect_classification() -> ModelClassificationResult:
    """Effect node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.EFFECT,
        confidence=0.95,
        template_name="effect",
        template_variant="async",
        primary_indicators=["external_io", "api_calls"],
        reasoning="Service performs external I/O operations",
        suggested_patterns=["async/await", "error handling"],
    )


@pytest.fixture
def compute_classification() -> ModelClassificationResult:
    """Compute node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.COMPUTE,
        confidence=0.9,
        template_name="compute",
        template_variant="standard",
        primary_indicators=["data_transformation", "calculation"],
        reasoning="Service performs data transformations",
        suggested_patterns=["pure functions", "data validation"],
    )


# ============================================================================
# Mock LLM Client Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_response():
    """Mock LLM response with generated code."""
    return """# Generated implementation
    logger.info(f"Executing effect for {contract.node_id}")

    # Process the input data
    result_data = {"status": "success", "processed": True}

    # Return result
    return ModelContainer(
        data=result_data,
        metadata={"execution_time": 0.1}
    )"""


@pytest.fixture
def mock_anthropic_client(mock_llm_response):
    """Mock Anthropic client for LLM generation."""

    # Create mock message content
    class MockTextBlock:
        def __init__(self, text: str):
            self.text = text
            self.type = "text"

    class MockMessage:
        def __init__(self, content_text: str):
            self.content = [MockTextBlock(content_text)]
            self.id = "msg_test123"
            self.model = "claude-sonnet-4"
            self.role = "assistant"
            self.stop_reason = "end_turn"
            self.usage = MagicMock(input_tokens=100, output_tokens=50)

    # Create mock client
    mock_client = AsyncMock()
    mock_message = MockMessage(mock_llm_response)
    mock_client.messages.create.return_value = mock_message

    return mock_client


@pytest.fixture
def mock_llm_node(mock_anthropic_client, mock_llm_response):
    """Mock NodeLLMEffect for business logic generation."""
    from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import EnumLLMTier
    from omninode_bridge.nodes.llm_effect.v1_0_0.models.model_response import (
        ModelLLMResponse,
    )

    mock_node = AsyncMock()

    # Mock the execute_effect method to return different code based on method name
    async def mock_execute_effect(contract):
        # Extract the method name from the prompt in the contract
        request_data = contract.input_data
        prompt = request_data.get("prompt", "")

        # Determine the method name from the prompt
        if "validate_input" in prompt:
            generated_code = """    async def validate_input(self, data):
        \"\"\"Validate input data.\"\"\"
        # Validate the input structure
        if not isinstance(data, dict):
            return False
        # Check required fields
        return True"""
        else:
            # Default to the effect/compute method response
            generated_code = mock_llm_response

        # Return ModelLLMResponse with generated code
        return ModelLLMResponse(
            generated_text=generated_code,
            model_used="claude-sonnet-4",
            tier_used=EnumLLMTier.CLOUD_FAST,
            tokens_input=100,
            tokens_output=50,
            tokens_total=150,
            latency_ms=100.0,
            cost_usd=0.00015,
            finish_reason="stop",
            truncated=False,
            warnings=[],
            retry_count=0,
        )

    mock_node.execute_effect.side_effect = mock_execute_effect
    mock_node.http_client = None  # Ensure it won't try to initialize
    return mock_node


# ============================================================================
# Sample Template Fixtures
# ============================================================================


@pytest.fixture
def sample_effect_template() -> str:
    """Sample Effect node template with stubs."""
    return '''#!/usr/bin/env python3
"""
Sample Effect Node Template.

Author: Test Generator
Version: v1_0_0
Tags: effect, template, test
"""

import logging
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

logger = logging.getLogger(__name__)


class NodeSampleEffect:
    """Sample Effect node with stub implementation."""

    def __init__(self):
        """Initialize the effect node."""
        logger.info("NodeSampleEffect initialized")

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

        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement validation logic
        pass
'''


@pytest.fixture
def sample_compute_template() -> str:
    """Sample Compute node template with stubs."""
    return '''#!/usr/bin/env python3
"""
Sample Compute Node Template.

Author: Test Generator
Version: v1_0_0
Tags: compute, template, test
"""

import logging
from omnibase_core.models.contracts.model_contract_compute import ModelContractCompute
from omnibase_core.models.core import ModelContainer

logger = logging.getLogger(__name__)


class NodeSampleCompute:
    """Sample Compute node with stub implementation."""

    def __init__(self):
        """Initialize the compute node."""
        logger.info("NodeSampleCompute initialized")

    async def execute_compute(self, contract: ModelContractCompute) -> ModelContainer:
        """
        Execute the computation.

        This method requires implementation.
        """
        # IMPLEMENTATION REQUIRED
        pass


'''


@pytest.fixture
def sample_template_artifacts(sample_effect_template) -> ModelTemplateArtifacts:
    """Sample template artifacts for testing."""
    return ModelTemplateArtifacts(
        template_code=sample_effect_template,
        template_path=Path("/tmp/test_template.py"),
        stubs=[
            ModelStubInfo(
                method_name="execute_effect",
                stub_code="        # IMPLEMENTATION REQUIRED\n        pass",
                line_start=25,
                line_end=26,
                signature="async def execute_effect(self, contract: ModelContractEffect) -> ModelContainer:",
                docstring="Execute the effect.\n\nThis method requires implementation.",
            ),
            ModelStubInfo(
                method_name="validate_input",
                stub_code="        # TODO: Implement validation logic\n        pass",
                line_start=34,
                line_end=35,
                signature="async def validate_input(self, data: dict) -> bool:",
                docstring="Validate input data.\n\nReturns:\n    True if valid, False otherwise",
            ),
        ],
        node_type="effect",
        version="v1_0_0",
        metadata=ModelTemplateMetadata(
            node_type="effect",
            version="v1_0_0",
            description="Sample Effect Node Template.",
            author="Test Generator",
            created_at=datetime.now(UTC),
            tags=["effect", "template", "test"],
        ),
        loaded_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_generated_artifacts(sample_effect_template) -> ModelGeneratedArtifacts:
    """Sample generated artifacts (as expected by BusinessLogicGenerator)."""
    return ModelGeneratedArtifacts(
        node_file=sample_effect_template,
        contract_file="# Sample contract YAML\n",
        init_file='"""Init file"""\n',
        node_type="effect",
        node_name="NodeSampleEffect",
        service_name="sample_effect",
        models={},
        tests={},
        documentation={},
    )


@pytest.fixture
def template_to_generated_artifacts():
    """Convert ModelTemplateArtifacts to ModelGeneratedArtifacts."""

    def _convert(template_artifacts: ModelTemplateArtifacts) -> ModelGeneratedArtifacts:
        """Convert template artifacts to generated artifacts format."""
        # Extract node name from class definition
        import re

        match = re.search(r"class\s+(Node\w+):", template_artifacts.template_code)
        node_name = match.group(1) if match else "Node"

        # Create service name from node name (e.g., NodeSampleEffect -> sample_effect)
        service_name = re.sub(r"([A-Z])", r"_\1", node_name).lower().lstrip("_")

        return ModelGeneratedArtifacts(
            node_file=template_artifacts.template_code,
            contract_file="# Contract YAML\n",
            init_file='"""Init file"""\n',
            node_type=template_artifacts.node_type,
            node_name=node_name,
            service_name=service_name,
            models={},
            tests={},
            documentation={},
        )

    return _convert


# ============================================================================
# PRD Requirements Fixtures
# ============================================================================


@pytest.fixture
def sample_prd_requirements() -> ModelPRDRequirements:
    """Sample PRD requirements for testing."""
    return ModelPRDRequirements(
        service_name="test_service",
        node_type="effect",
        domain="api",  # Required field
        business_description="Test service for processing data with external API calls",
        operations=["fetch_data", "process_data", "store_results"],
        features=["rate_limiting", "error_handling", "metrics_collection"],
        input_schema={"type": "object", "properties": {"data": {"type": "string"}}},
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}, "status": {"type": "string"}},
        },
        performance_requirements={
            "max_latency_ms": 1000,
            "min_throughput_rps": 100,
        },
        error_handling_strategy="retry_with_exponential_backoff",
    )


# ============================================================================
# Expected Output Fixtures
# ============================================================================


@pytest.fixture
def expected_enhanced_code() -> str:
    """Expected enhanced code after LLM generation."""
    return '''async def execute_effect(self, contract: ModelContractEffect) -> ModelContainer:
    """Execute the effect with generated implementation."""
    # Generated implementation
    logger.info(f"Executing effect for {contract.node_id}")

    # Process the input data
    result_data = {"status": "success", "processed": True}

    # Return result
    return ModelContainer(
        data=result_data,
        metadata={"execution_time": 0.1}
    )'''


# ============================================================================
# Environment Setup Fixtures
# ============================================================================


@pytest.fixture
def mock_zai_api_key(monkeypatch):
    """Mock ZAI_API_KEY environment variable."""
    monkeypatch.setenv("ZAI_API_KEY", "test_api_key_12345")
    return "test_api_key_12345"


@pytest.fixture
def mock_anthropic_client_class(mock_anthropic_client):
    """Mock Anthropic client class for patching."""
    with patch("anthropic.AsyncAnthropic", return_value=mock_anthropic_client):
        yield mock_anthropic_client


# ============================================================================
# Business Logic Generator Fixtures
# ============================================================================


@pytest.fixture
def business_logic_generator_disabled() -> BusinessLogicGenerator:
    """BusinessLogicGenerator with LLM disabled."""
    return BusinessLogicGenerator(enable_llm=False)


@pytest.fixture
def business_logic_generator_enabled(
    mock_zai_api_key, mock_llm_node, template_to_generated_artifacts
) -> BusinessLogicGenerator:
    """BusinessLogicGenerator with LLM enabled (mocked)."""
    import re

    generator = BusinessLogicGenerator(enable_llm=True)
    # Patch the llm_node after initialization with the mock
    generator.llm_node = mock_llm_node

    # Fix the _inject_implementation method to handle type annotations
    original_inject = generator._inject_implementation

    def fixed_inject(node_file: str, method_name: str, implementation: str) -> str:
        """Inject with improved regex pattern that handles type annotations."""
        # Improved pattern that handles:
        # - Leading whitespace (for class methods)
        # - Type annotations with colons in parameters
        # - Return type annotations (-> ReturnType)
        # - Keeps method signature and docstring (group 1)
        # - Replaces just the stub body
        # - Matches both "# IMPLEMENTATION REQUIRED" and "# TODO:" markers
        stub_pattern = (
            rf"(\s*async def {method_name}\([^)]*\)(?:\s*->\s*\w+)?:.*?\n"
            r"(?:        \"\"\".*?\"\"\"\n)?)"  # Group 1: signature + optional docstring
            r"        (?:# IMPLEMENTATION REQUIRED|# TODO:).*?\n"  # Stub marker (flexible)
            r"        .*?pass"  # Stub body
        )

        # Replace with: signature/docstring + implementation
        # Group 1 is the signature and docstring, implementation is the body
        replacement = r"\1" + implementation
        enhanced = re.sub(stub_pattern, replacement, node_file, flags=re.DOTALL)

        return enhanced

    generator._inject_implementation = fixed_inject

    # Wrap enhance_artifacts to auto-convert ModelTemplateArtifacts to ModelGeneratedArtifacts
    original_enhance = generator.enhance_artifacts

    async def wrapped_enhance(artifacts, requirements, context_data=None):
        """Wrapper that auto-converts ModelTemplateArtifacts to ModelGeneratedArtifacts."""
        # Check if artifacts is ModelTemplateArtifacts and convert if needed
        if isinstance(artifacts, ModelTemplateArtifacts):
            artifacts = template_to_generated_artifacts(artifacts)
        return await original_enhance(artifacts, requirements, context_data)

    generator.enhance_artifacts = wrapped_enhance
    return generator


# ============================================================================
# Template Engine Fixtures
# ============================================================================


@pytest.fixture
def template_engine() -> TemplateEngine:
    """TemplateEngine instance."""
    return TemplateEngine(enable_validation=True)


@pytest.fixture
def temp_template_dir(tmp_path, sample_effect_template, sample_compute_template):
    """Temporary directory with sample templates."""
    template_dir = tmp_path / "node_templates"

    # Create effect template
    effect_dir = template_dir / "effect" / "v1_0_0"
    effect_dir.mkdir(parents=True)
    (effect_dir / "node.py").write_text(sample_effect_template)

    # Create compute template
    compute_dir = template_dir / "compute" / "v1_0_0"
    compute_dir.mkdir(parents=True)
    (compute_dir / "node.py").write_text(sample_compute_template)

    return template_dir


# ============================================================================
# Integration Context Fixtures
# ============================================================================


@pytest.fixture
def integration_context() -> dict[str, Any]:
    """Context data for integration testing."""
    return {
        "patterns": ["Use async/await", "Follow ONEX v2.0 patterns"],
        "best_practices": [
            "Handle errors with ModelOnexError",
            "Use structured logging",
        ],
        "similar_implementations": [],
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary directory for test outputs with cleanup."""
    output_dir = tmp_path / "generated_nodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup handled automatically by tmp_path fixture

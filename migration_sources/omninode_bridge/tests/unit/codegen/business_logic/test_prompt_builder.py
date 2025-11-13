#!/usr/bin/env python3
"""
Unit tests for PromptBuilder.

Tests prompt construction, context gathering, template formatting,
and integration points for LLM business logic generation.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"


def load_module_from_file(module_name: str, file_path: Path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load pydantic first to satisfy dependencies
try:
    import pydantic  # noqa: F401
except ImportError:
    pytest.skip("Pydantic not available", allow_module_level=True)

# Load models module and register it with the expected import path
models_module = load_module_from_file(
    "omninode_bridge.codegen.business_logic.models",
    SRC_ROOT / "omninode_bridge" / "codegen" / "business_logic" / "models.py",
)

# Load prompt_builder module (it can now import models using absolute import)
prompt_builder_module = load_module_from_file(
    "omninode_bridge.codegen.business_logic.prompt_builder",
    SRC_ROOT / "omninode_bridge" / "codegen" / "business_logic" / "prompt_builder.py",
)

# Extract classes for use in tests
GenerationContext = models_module.GenerationContext
PromptPair = models_module.PromptPair
StubInfo = models_module.StubInfo
PromptBuilder = prompt_builder_module.PromptBuilder


@pytest.fixture
def sample_context():
    """Sample generation context."""
    return GenerationContext(
        node_type="effect",
        service_name="postgres_client",
        business_description="PostgreSQL database client for executing queries",
        operations=["execute_query", "execute_transaction", "check_connection"],
        features=["connection pooling", "retry logic", "async operations"],
        contract_spec={
            "input": {
                "query": {
                    "type": "str",
                    "description": "SQL query to execute",
                    "required": True,
                },
                "params": {
                    "type": "dict",
                    "description": "Query parameters",
                    "required": False,
                },
            },
            "output": {
                "result": {
                    "type": "list",
                    "description": "Query results",
                },
                "rows_affected": {
                    "type": "int",
                    "description": "Number of rows affected",
                },
            },
            "constraints": [
                "Must use parameterized queries",
                "Must handle connection errors",
            ],
        },
        performance_requirements={
            "max_latency_ms": 100,
            "max_connections": 50,
        },
    )


@pytest.fixture
def sample_stub_info():
    """Sample stub information."""
    return StubInfo(
        file_path="/path/to/node.py",
        method_name="execute_effect",
        stub_code='        """Execute database query."""\n        pass',
        line_start=42,
        line_end=44,
        signature="async def execute_effect(self, contract: ModelContract) -> ModelContract:",
        docstring="Execute database query with connection pooling and retry logic.",
    )


class TestPromptBuilder:
    """Test suite for PromptBuilder."""

    def test_init_default_templates(self):
        """Test initialization with default templates."""
        builder = PromptBuilder()

        assert builder.rag_client is None
        assert builder.kb_client is None
        assert builder.templates_dir.exists()
        assert len(builder.system_template) > 0
        assert len(builder.user_template) > 0

    def test_init_custom_clients(self):
        """Test initialization with custom RAG and KB clients."""
        mock_rag = MagicMock()
        mock_kb = MagicMock()

        builder = PromptBuilder(rag_client=mock_rag, kb_client=mock_kb)

        assert builder.rag_client == mock_rag
        assert builder.kb_client == mock_kb

    def test_template_loading(self):
        """Test template loading from files."""
        builder = PromptBuilder()

        # Check system template loaded
        assert "ONEX" in builder.system_template
        assert "production-ready" in builder.system_template
        assert "{node_type}" in builder.system_template

        # Check user template loaded
        assert "Service Name" in builder.user_template
        assert "{service_name}" in builder.user_template
        assert "{stub_code}" in builder.user_template

    @pytest.mark.asyncio
    async def test_build_prompt_basic(self, sample_context, sample_stub_info):
        """Test basic prompt building without RAG."""
        builder = PromptBuilder()

        prompts = await builder.build_prompt(sample_context, sample_stub_info)

        # Check prompt pair returned
        assert prompts.system_prompt
        assert prompts.user_prompt
        assert prompts.estimated_tokens > 0

        # Check system prompt contains node type
        assert "Effect" in prompts.system_prompt

        # Check user prompt contains context
        assert "postgres_client" in prompts.user_prompt
        assert "execute_query" in prompts.user_prompt
        assert "connection pooling" in prompts.user_prompt

    @pytest.mark.asyncio
    async def test_build_prompt_adds_best_practices(
        self, sample_context, sample_stub_info
    ):
        """Test that ONEX best practices are added based on node type."""
        builder = PromptBuilder()

        prompts = await builder.build_prompt(sample_context, sample_stub_info)

        # Check Effect node best practices are included
        assert "async/await" in prompts.user_prompt.lower()
        assert "circuit breaker" in prompts.user_prompt.lower()

    @pytest.mark.asyncio
    async def test_build_prompt_adds_error_handling_patterns(
        self, sample_context, sample_stub_info
    ):
        """Test that error handling patterns are added."""
        builder = PromptBuilder()

        prompts = await builder.build_prompt(sample_context, sample_stub_info)

        # Check error handling patterns included
        assert "ModelOnexError" in prompts.user_prompt
        assert "correlation_id" in prompts.user_prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_rag_client(self, sample_context, sample_stub_info):
        """Test prompt building with RAG client for similar patterns."""
        # Mock RAG client that returns similar patterns
        mock_rag = AsyncMock()
        builder = PromptBuilder(rag_client=mock_rag)

        # Mock the _gather_similar_patterns method
        with patch.object(
            builder,
            "_gather_similar_patterns",
            return_value=[
                "Pattern 1: Use connection pooling",
                "Pattern 2: Retry logic",
            ],
        ):
            prompts = await builder.build_prompt(sample_context, sample_stub_info)

            # Similar patterns should be in context now
            assert sample_context.similar_patterns

    @pytest.mark.asyncio
    async def test_gather_similar_patterns_no_client(self):
        """Test pattern gathering without RAG client."""
        builder = PromptBuilder()

        patterns = await builder._gather_similar_patterns("effect", "execute_effect")

        assert patterns == []

    @pytest.mark.asyncio
    async def test_gather_similar_patterns_with_client(self):
        """Test pattern gathering with RAG client."""
        mock_rag = AsyncMock()
        builder = PromptBuilder(rag_client=mock_rag)

        # Should not raise error even though not implemented
        patterns = await builder._gather_similar_patterns("effect", "execute_effect")

        # Currently returns empty list (TODO: implement RAG integration)
        assert isinstance(patterns, list)

    def test_format_contract_spec_full(self, sample_context):
        """Test contract specification formatting with full spec."""
        builder = PromptBuilder()

        formatted = builder._format_contract_spec(sample_context.contract_spec)

        # Check input fields formatted
        assert "Input Fields" in formatted
        assert "query: str (required)" in formatted
        assert "SQL query to execute" in formatted

        # Check output fields formatted
        assert "Output Fields" in formatted
        assert "result: list" in formatted

        # Check constraints formatted
        assert "Constraints" in formatted
        assert "parameterized queries" in formatted

    def test_format_contract_spec_empty(self):
        """Test contract specification formatting with empty spec."""
        builder = PromptBuilder()

        formatted = builder._format_contract_spec({})

        assert formatted == "No contract specification provided"

    def test_format_list(self):
        """Test list formatting as bullet points."""
        builder = PromptBuilder()

        # Non-empty list
        formatted = builder._format_list(["Item 1", "Item 2", "Item 3"])
        assert formatted == "- Item 1\n- Item 2\n- Item 3"

        # Empty list
        formatted = builder._format_list([], "No items")
        assert formatted == "No items"

    def test_format_dict(self):
        """Test dictionary formatting as key-value pairs."""
        builder = PromptBuilder()

        # Non-empty dict
        formatted = builder._format_dict({"key1": "value1", "key2": 42})
        assert "- key1: value1" in formatted
        assert "- key2: 42" in formatted

        # Empty dict
        formatted = builder._format_dict({}, "No data")
        assert formatted == "No data"

    def test_estimate_tokens(self):
        """Test token estimation."""
        builder = PromptBuilder()

        # Test with sample text
        text = "This is a test prompt with approximately 100 characters for token estimation testing purposes here."
        tokens = builder.estimate_tokens(text)

        # Should be roughly text length / 4
        assert tokens > 0
        assert tokens == len(text) // 4

    @pytest.mark.asyncio
    async def test_build_prompt_all_node_types(self, sample_stub_info):
        """Test prompt building for all node types."""
        builder = PromptBuilder()

        node_types = ["effect", "compute", "reducer", "orchestrator"]

        for node_type in node_types:
            context = GenerationContext(
                node_type=node_type,
                service_name="test_service",
                business_description="Test service",
                operations=["test_operation"],
                features=["test_feature"],
            )

            prompts = await builder.build_prompt(context, sample_stub_info)

            # Check node type-specific best practices included
            assert prompts.user_prompt
            assert prompts.system_prompt

            # Verify node type in system prompt
            assert node_type.capitalize() in prompts.system_prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_performance_requirements(
        self, sample_context, sample_stub_info
    ):
        """Test that performance requirements are included in prompt."""
        builder = PromptBuilder()

        prompts = await builder.build_prompt(sample_context, sample_stub_info)

        # Check performance requirements in prompt
        assert "max_latency_ms" in prompts.user_prompt
        assert "100" in prompts.user_prompt

    @pytest.mark.asyncio
    async def test_build_prompt_includes_stub_code(
        self, sample_context, sample_stub_info
    ):
        """Test that stub code is included for context."""
        builder = PromptBuilder()

        prompts = await builder.build_prompt(sample_context, sample_stub_info)

        # Check stub code included
        assert sample_stub_info.stub_code in prompts.user_prompt

    @pytest.mark.asyncio
    async def test_build_prompt_includes_method_signature(
        self, sample_context, sample_stub_info
    ):
        """Test that method signature is included."""
        builder = PromptBuilder()

        prompts = await builder.build_prompt(sample_context, sample_stub_info)

        # Check signature included
        assert "execute_effect" in prompts.user_prompt
        assert "ModelContract" in prompts.user_prompt

    @pytest.mark.asyncio
    async def test_build_prompt_handles_missing_optional_fields(self, sample_stub_info):
        """Test prompt building with minimal context."""
        builder = PromptBuilder()

        # Minimal context
        context = GenerationContext(
            node_type="compute",
            service_name="minimal_service",
            business_description="Minimal test",
        )

        prompts = await builder.build_prompt(context, sample_stub_info)

        # Should still generate valid prompts
        assert prompts.system_prompt
        assert prompts.user_prompt
        assert prompts.estimated_tokens > 0

    def test_onex_best_practices_coverage(self):
        """Test that ONEX best practices are defined for all node types."""
        builder = PromptBuilder()

        required_node_types = ["effect", "compute", "reducer", "orchestrator"]

        for node_type in required_node_types:
            assert node_type in builder.ONEX_BEST_PRACTICES
            assert len(builder.ONEX_BEST_PRACTICES[node_type]) > 0

    def test_error_handling_patterns_defined(self):
        """Test that error handling patterns are defined."""
        builder = PromptBuilder()

        assert len(builder.ERROR_HANDLING_PATTERNS) > 0
        assert any(
            "ModelOnexError" in pattern for pattern in builder.ERROR_HANDLING_PATTERNS
        )

#!/usr/bin/env python3
"""
Pytest configuration and fixtures for unit tests.

Provides fixtures for mocking LLM calls, templates, and other dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add fixtures to path
FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "codegen"
if str(FIXTURES_PATH) not in sys.path:
    sys.path.insert(0, str(FIXTURES_PATH))

from mock_responses import (
    MOCK_LLM_RESPONSE_COMPLEX,
    MOCK_LLM_RESPONSE_MODERATE,
    MOCK_LLM_RESPONSE_SIMPLE,
    SAMPLE_CODE_WITH_SECURITY_ISSUES,
    SAMPLE_CODE_WITH_STUBS,
    SAMPLE_CODE_WITH_SYNTAX_ERROR,
    SAMPLE_VALID_CODE,
)
from sample_requirements import (
    get_complex_orchestration_requirements,
    get_invalid_requirements,
    get_moderate_complexity_requirements,
    get_reducer_requirements,
    get_simple_crud_requirements,
)

# Optional imports - gracefully handle missing dependencies (e.g., omnibase_core)
try:
    from omninode_bridge.codegen.node_classifier import (
        EnumNodeType,
        ModelClassificationResult,
    )
    from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

    CODEGEN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Dependencies not available - fixtures will be skipped
    CODEGEN_AVAILABLE = False
    EnumNodeType = None  # type: ignore
    ModelClassificationResult = None  # type: ignore
    ModelPRDRequirements = None  # type: ignore

# ============================================================================
# Requirements Fixtures
# ============================================================================


@pytest.fixture
def simple_crud_requirements() -> ModelPRDRequirements:
    """Simple CRUD requirements (low complexity)."""
    if not CODEGEN_AVAILABLE:
        pytest.skip("codegen dependencies not available")
    return get_simple_crud_requirements()


@pytest.fixture
def moderate_complexity_requirements() -> ModelPRDRequirements:
    """Moderate complexity requirements."""
    return get_moderate_complexity_requirements()


@pytest.fixture
def complex_orchestration_requirements() -> ModelPRDRequirements:
    """Complex orchestration requirements (high complexity)."""
    return get_complex_orchestration_requirements()


@pytest.fixture
def reducer_requirements() -> ModelPRDRequirements:
    """Reducer node requirements."""
    return get_reducer_requirements()


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
        reasoning="Service performs external I/O operations",
        suggested_patterns=["async/await", "error handling"],
        template_name="effect_standard",
    )


@pytest.fixture
def compute_classification() -> ModelClassificationResult:
    """Compute node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.COMPUTE,
        confidence=0.9,
        reasoning="Service performs data transformations",
        suggested_patterns=["pure functions", "data validation"],
        template_name="compute_standard",
    )


@pytest.fixture
def orchestrator_classification() -> ModelClassificationResult:
    """Orchestrator node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.ORCHESTRATOR,
        confidence=0.92,
        reasoning="Service coordinates multiple operations",
        suggested_patterns=["workflow management", "error aggregation"],
        template_name="orchestrator_standard",
    )


@pytest.fixture
def reducer_classification() -> ModelClassificationResult:
    """Reducer node classification."""
    return ModelClassificationResult(
        node_type=EnumNodeType.REDUCER,
        confidence=0.88,
        reasoning="Service aggregates streaming data",
        suggested_patterns=["state management", "windowing"],
        template_name="reducer_standard",
    )


# ============================================================================
# Mock LLM Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_simple_response():
    """Mock LLM response for simple logic."""
    return MOCK_LLM_RESPONSE_SIMPLE


@pytest.fixture
def mock_llm_moderate_response():
    """Mock LLM response for moderate complexity."""
    return MOCK_LLM_RESPONSE_MODERATE


@pytest.fixture
def mock_llm_complex_response():
    """Mock LLM response for complex logic."""
    return MOCK_LLM_RESPONSE_COMPLEX


@pytest.fixture
def mock_llm_node():
    """Mock NodeLLMEffect for business logic generation."""
    if not CODEGEN_AVAILABLE:
        pytest.skip("codegen dependencies not available")

    try:
        from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import (
            EnumLLMTier,
        )
        from omninode_bridge.nodes.llm_effect.v1_0_0.models.model_response import (
            ModelLLMResponse,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("LLM effect node dependencies not available")

    mock_node = AsyncMock()

    async def mock_execute_effect(contract):
        # Return mock LLM response
        return ModelLLMResponse(
            generated_text=MOCK_LLM_RESPONSE_SIMPLE,
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
    return mock_node


@pytest.fixture
def mock_zai_api_key(monkeypatch):
    """Mock ZAI_API_KEY environment variable."""
    monkeypatch.setenv("ZAI_API_KEY", "test_api_key_12345")
    return "test_api_key_12345"


# ============================================================================
# Sample Code Fixtures
# ============================================================================


@pytest.fixture
def sample_valid_code():
    """Sample valid generated code (no stubs)."""
    return SAMPLE_VALID_CODE


@pytest.fixture
def sample_code_with_stubs():
    """Sample code with stubs (for injection testing)."""
    return SAMPLE_CODE_WITH_STUBS


@pytest.fixture
def sample_code_with_syntax_error():
    """Sample code with syntax errors."""
    return SAMPLE_CODE_WITH_SYNTAX_ERROR


@pytest.fixture
def sample_code_with_security_issues():
    """Sample code with security issues."""
    return SAMPLE_CODE_WITH_SECURITY_ISSUES


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for generated code."""
    output_dir = tmp_path / "generated_nodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Phase 3 Fixtures
# ============================================================================


@pytest.fixture
def mock_pattern_library():
    """Mock pattern library for testing."""
    return {
        "circuit_breaker": {
            "name": "circuit_breaker",
            "category": "resilience",
            "node_types": ["effect", "orchestrator"],
            "complexity": 5,
            "description": "Circuit breaker pattern for fault tolerance",
            "code_template": "# Circuit breaker implementation",
            "dependencies": [],
        },
        "retry_logic": {
            "name": "retry_logic",
            "category": "resilience",
            "node_types": ["effect"],
            "complexity": 3,
            "description": "Retry logic with exponential backoff",
            "code_template": "# Retry implementation",
            "dependencies": [],
        },
        "health_checks": {
            "name": "health_checks",
            "category": "monitoring",
            "node_types": ["effect", "compute", "orchestrator", "reducer"],
            "complexity": 2,
            "description": "Health check endpoints",
            "code_template": "# Health check implementation",
            "dependencies": [],
        },
        "metrics_collection": {
            "name": "metrics_collection",
            "category": "monitoring",
            "node_types": ["effect", "compute", "orchestrator", "reducer"],
            "complexity": 3,
            "description": "Metrics collection and reporting",
            "code_template": "# Metrics implementation",
            "dependencies": [],
        },
        "connection_pooling": {
            "name": "connection_pooling",
            "category": "performance",
            "node_types": ["effect"],
            "complexity": 4,
            "description": "Database connection pooling",
            "code_template": "# Connection pool implementation",
            "dependencies": ["asyncpg"],
        },
    }


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns canned responses without API calls."""
    mock_client = AsyncMock()

    async def mock_generate(*args, **kwargs):
        return MOCK_LLM_RESPONSE_SIMPLE

    mock_client.generate.side_effect = mock_generate
    mock_client.generate_text.side_effect = mock_generate
    return mock_client


@pytest.fixture
def sample_variant_metadata():
    """Sample variant metadata for testing."""
    return {
        "database_heavy": {
            "name": "database_heavy",
            "description": "Template optimized for database-heavy operations",
            "selection_criteria": {
                "features": ["connection_pooling", "transactions"],
                "operations": ["create", "read", "update", "delete"],
                "min_score": 0.7,
            },
            "template_path": "templates/variants/database_heavy.j2",
        },
        "api_heavy": {
            "name": "api_heavy",
            "description": "Template optimized for API-heavy operations",
            "selection_criteria": {
                "features": ["http_client", "retry_logic"],
                "operations": ["fetch", "post", "put"],
                "min_score": 0.7,
            },
            "template_path": "templates/variants/api_heavy.j2",
        },
    }


@pytest.fixture
def sample_enhanced_contract():
    """Sample enhanced contract with subcontracts."""
    return {
        "node_id": "test_node_effect",
        "node_type": "effect",
        "version": "v1_0_0",
        "metadata": {
            "service_name": "test_service",
            "domain": "test",
        },
        "subcontracts": {
            "fsm": {
                "states": ["idle", "processing", "completed", "failed"],
                "initial_state": "idle",
                "transitions": [
                    {
                        "from": "idle",
                        "to": "processing",
                        "trigger": "start",
                    },
                    {
                        "from": "processing",
                        "to": "completed",
                        "trigger": "finish",
                    },
                ],
            },
            "event_type": {
                "events": [
                    {"name": "processing_started", "schema": {}},
                    {"name": "processing_completed", "schema": {}},
                ],
            },
        },
    }


@pytest.fixture
def sample_mixin_scores():
    """Sample mixin scores for testing."""
    return {
        "health_check": {"relevance": 0.9, "complexity": 0.8, "total": 0.85},
        "metrics": {"relevance": 0.85, "complexity": 0.7, "total": 0.775},
        "circuit_breaker": {"relevance": 0.7, "complexity": 0.6, "total": 0.65},
        "retry_logic": {"relevance": 0.8, "complexity": 0.7, "total": 0.75},
    }


@pytest.fixture
def sample_llm_context():
    """Sample LLM context for testing."""
    return {
        "requirements": {
            "service_name": "test_service",
            "node_type": "effect",
            "operations": ["process"],
        },
        "patterns": [
            {
                "name": "circuit_breaker",
                "description": "Fault tolerance pattern",
            }
        ],
        "constraints": {
            "max_complexity": 10,
            "required_patterns": ["health_checks"],
        },
        "examples": [
            {
                "name": "simple_effect",
                "code": "# Example code",
            }
        ],
    }


@pytest.fixture
def mock_pattern_matcher():
    """Mock pattern matcher for testing."""
    matcher = MagicMock()
    matcher.match_patterns.return_value = ["circuit_breaker", "health_checks"]
    matcher.score_pattern.return_value = 0.8
    return matcher


@pytest.fixture
def mock_mixin_recommender():
    """Mock mixin recommender for testing."""
    recommender = MagicMock()
    recommender.recommend.return_value = [
        "health_check",
        "metrics",
    ]
    recommender.score_mixin.return_value = 0.85
    return recommender


@pytest.fixture
def mock_contract_parser():
    """Mock contract parser for testing."""
    parser = MagicMock()
    parser.parse.return_value = {
        "node_id": "test_node",
        "subcontracts": {},
    }
    return parser

"""
Performance tests for AI Quorum.

Tests validate 2-10s latency target for quorum validation.
"""

import time

import pytest

from omninode_bridge.agents.workflows.ai_quorum import AIQuorum
from omninode_bridge.agents.workflows.llm_client import MockLLMClient
from omninode_bridge.agents.workflows.quorum_models import (
    ModelConfig,
    ValidationContext,
)


@pytest.fixture
def performance_model_configs():
    """Model configurations for performance testing."""
    return [
        ModelConfig(
            model_id="gemini",
            model_name="gemini-1.5-pro",
            weight=2.0,
            endpoint="http://mock-gemini",
            api_key_env="MOCK_KEY",
        ),
        ModelConfig(
            model_id="glm-4.5",
            model_name="glm-4-plus",
            weight=2.0,
            endpoint="http://mock-glm",
            api_key_env="MOCK_KEY",
        ),
        ModelConfig(
            model_id="glm-air",
            model_name="glm-4-air",
            weight=1.5,
            endpoint="http://mock-glm",
            api_key_env="MOCK_KEY",
        ),
        ModelConfig(
            model_id="codestral",
            model_name="codestral-latest",
            weight=1.0,
            endpoint="http://mock-codestral",
            api_key_env="MOCK_KEY",
        ),
    ]


@pytest.fixture
def validation_context():
    """Validation context for performance testing."""
    return ValidationContext(
        node_type="effect",
        contract_summary="Performance test contract",
        validation_criteria=[
            "ONEX v2.0 compliance",
            "Code quality",
            "Security",
        ],
    )


@pytest.mark.asyncio
async def test_quorum_latency_parallel_execution(
    performance_model_configs, validation_context
):
    """
    Test quorum latency meets 2-10s target with parallel execution.

    With 4 models and 1000ms latency each:
    - Sequential: ~4000ms
    - Parallel: ~1000ms (all models called simultaneously)

    Target: <2000ms for this test (models have 1000ms latency)
    """
    quorum = AIQuorum(
        model_configs=performance_model_configs,
        pass_threshold=0.6,
        min_participating_weight=3.0,
    )

    # Register mock clients with 1000ms latency each
    for config in performance_model_configs:
        client = MockLLMClient(
            model_id=config.model_id,
            model_name=config.model_name,
            default_vote=True,
            default_confidence=0.9,
            latency_ms=1000.0,  # 1 second per model
        )
        quorum.register_client(config.model_id, client)

    await quorum.initialize()

    # Measure quorum latency
    start = time.perf_counter()
    result = await quorum.validate_code(
        code="def test(): pass", context=validation_context
    )
    duration_ms = (time.perf_counter() - start) * 1000

    # Verify result
    assert result.passed is True
    assert len(result.votes) == 4

    # Verify parallel execution (should be ~1000ms, not 4000ms)
    # Allow some overhead (1500ms max for 1000ms latency)
    assert duration_ms < 1500, f"Expected <1500ms, got {duration_ms:.1f}ms"

    # Verify result.duration_ms matches actual duration (within 200ms)
    assert abs(result.duration_ms - duration_ms) < 200

    await quorum.close()


@pytest.mark.asyncio
async def test_quorum_latency_fast_models(performance_model_configs, validation_context):
    """
    Test quorum latency with fast models (100ms each).

    Target: <500ms total (100ms model + overhead)
    """
    quorum = AIQuorum(
        model_configs=performance_model_configs,
        pass_threshold=0.6,
        min_participating_weight=3.0,
    )

    # Register mock clients with 100ms latency each (fast models)
    for config in performance_model_configs:
        client = MockLLMClient(
            model_id=config.model_id,
            model_name=config.model_name,
            default_vote=True,
            default_confidence=0.9,
            latency_ms=100.0,  # 100ms per model
        )
        quorum.register_client(config.model_id, client)

    await quorum.initialize()

    # Measure quorum latency
    start = time.perf_counter()
    result = await quorum.validate_code(
        code="def test(): pass", context=validation_context
    )
    duration_ms = (time.perf_counter() - start) * 1000

    # Verify fast execution
    assert duration_ms < 500, f"Expected <500ms, got {duration_ms:.1f}ms"

    await quorum.close()


@pytest.mark.asyncio
async def test_quorum_latency_slow_models(performance_model_configs, validation_context):
    """
    Test quorum latency with slow models (2500ms each).

    Target: <10000ms (within 10s target for worst case)
    """
    quorum = AIQuorum(
        model_configs=performance_model_configs,
        pass_threshold=0.6,
        min_participating_weight=3.0,
    )

    # Register mock clients with 2500ms latency each (slow models)
    for config in performance_model_configs:
        client = MockLLMClient(
            model_id=config.model_id,
            model_name=config.model_name,
            default_vote=True,
            default_confidence=0.9,
            latency_ms=2500.0,  # 2.5 seconds per model
        )
        quorum.register_client(config.model_id, client)

    await quorum.initialize()

    # Measure quorum latency
    start = time.perf_counter()
    result = await quorum.validate_code(
        code="def test(): pass", context=validation_context
    )
    duration_ms = (time.perf_counter() - start) * 1000

    # Verify execution within 10s target
    assert duration_ms < 10000, f"Expected <10000ms, got {duration_ms:.1f}ms"

    # Verify parallel execution (should be ~2500ms, not 10000ms)
    assert duration_ms < 3500, f"Expected parallel <3500ms, got {duration_ms:.1f}ms"

    await quorum.close()


@pytest.mark.asyncio
async def test_quorum_performance_metrics(
    performance_model_configs, validation_context
):
    """Test quorum records performance metrics correctly."""
    quorum = AIQuorum(
        model_configs=performance_model_configs,
        pass_threshold=0.6,
        min_participating_weight=3.0,
    )

    # Register mock clients
    for config in performance_model_configs:
        client = MockLLMClient(
            model_id=config.model_id,
            model_name=config.model_name,
            default_vote=True,
            default_confidence=0.9,
            latency_ms=100.0,
        )
        quorum.register_client(config.model_id, client)

    await quorum.initialize()

    # Run multiple validations
    for _ in range(5):
        result = await quorum.validate_code(
            code="def test(): pass", context=validation_context
        )
        assert result.duration_ms > 0

    # Check statistics
    stats = quorum.get_statistics()
    assert stats["total_validations"] == 5
    assert stats["total_passes"] == 5
    assert stats["pass_rate"] == 1.0

    await quorum.close()


@pytest.mark.asyncio
async def test_quorum_speedup_vs_sequential(
    performance_model_configs, validation_context
):
    """
    Test quorum provides speedup vs sequential execution.

    With 4 models at 500ms each:
    - Sequential: 2000ms
    - Parallel: ~500ms
    - Speedup: ~4x
    """
    quorum = AIQuorum(
        model_configs=performance_model_configs,
        pass_threshold=0.6,
        min_participating_weight=3.0,
    )

    # Register mock clients with 500ms latency
    for config in performance_model_configs:
        client = MockLLMClient(
            model_id=config.model_id,
            model_name=config.model_name,
            default_vote=True,
            default_confidence=0.9,
            latency_ms=500.0,
        )
        quorum.register_client(config.model_id, client)

    await quorum.initialize()

    # Measure parallel execution
    start = time.perf_counter()
    result = await quorum.validate_code(
        code="def test(): pass", context=validation_context
    )
    parallel_duration_ms = (time.perf_counter() - start) * 1000

    # Sequential would be 4 * 500ms = 2000ms
    sequential_duration_ms = 2000.0

    # Calculate speedup
    speedup = sequential_duration_ms / parallel_duration_ms

    # Verify speedup is significant (at least 2x, ideally close to 4x)
    assert speedup >= 2.0, f"Expected speedup >=2x, got {speedup:.1f}x"

    # Verify parallel execution is close to single model latency
    assert parallel_duration_ms < 800, f"Expected <800ms, got {parallel_duration_ms:.1f}ms"

    await quorum.close()

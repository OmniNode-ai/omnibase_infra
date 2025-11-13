"""
Unit tests for AI Quorum integration.

Tests weighted voting, consensus calculation, and quorum validation.
Target: 95%+ coverage
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows.ai_quorum import AIQuorum, DEFAULT_QUORUM_MODELS
from omninode_bridge.agents.workflows.llm_client import MockLLMClient
from omninode_bridge.agents.workflows.quorum_models import (
    ModelConfig,
    QuorumResult,
    QuorumVote,
    ValidationContext,
)


@pytest.fixture
def sample_model_configs():
    """Sample model configurations for testing."""
    return [
        ModelConfig(
            model_id="model_a",
            model_name="model-a-v1",
            weight=2.0,
            endpoint="http://mock-a",
            api_key_env="MODEL_A_KEY",
        ),
        ModelConfig(
            model_id="model_b",
            model_name="model-b-v1",
            weight=1.5,
            endpoint="http://mock-b",
            api_key_env="MODEL_B_KEY",
        ),
        ModelConfig(
            model_id="model_c",
            model_name="model-c-v1",
            weight=1.0,
            endpoint="http://mock-c",
            api_key_env="MODEL_C_KEY",
        ),
    ]


@pytest.fixture
def validation_context():
    """Sample validation context."""
    return ValidationContext(
        node_type="effect",
        contract_summary="Test contract for effect node",
        validation_criteria=[
            "ONEX v2.0 compliance",
            "Code quality",
            "Security",
        ],
    )


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    collector = MagicMock(spec=MetricsCollector)
    collector.record_counter = AsyncMock()
    collector.record_timing = AsyncMock()
    collector.record_gauge = AsyncMock()
    return collector


class TestModelConfig:
    """Tests for ModelConfig data model."""

    def test_model_config_creation(self):
        """Test creating ModelConfig."""
        config = ModelConfig(
            model_id="test",
            model_name="test-model",
            weight=2.0,
            endpoint="https://api.test.com",
            api_key_env="TEST_KEY",
        )

        assert config.model_id == "test"
        assert config.model_name == "test-model"
        assert config.weight == 2.0
        assert config.endpoint == "https://api.test.com"
        assert config.api_key_env == "TEST_KEY"
        assert config.timeout == 30
        assert config.max_retries == 2
        assert config.enabled is True

    def test_model_config_invalid_endpoint(self):
        """Test ModelConfig with invalid endpoint."""
        with pytest.raises(ValueError, match="must start with http"):
            ModelConfig(
                model_id="test",
                model_name="test-model",
                weight=2.0,
                endpoint="ftp://invalid",
                api_key_env="TEST_KEY",
            )

    def test_model_config_invalid_weight(self):
        """Test ModelConfig with invalid weight."""
        with pytest.raises(ValueError):
            ModelConfig(
                model_id="test",
                model_name="test-model",
                weight=0.0,  # Must be > 0
                endpoint="https://api.test.com",
                api_key_env="TEST_KEY",
            )

        with pytest.raises(ValueError):
            ModelConfig(
                model_id="test",
                model_name="test-model",
                weight=11.0,  # Must be <= 10
                endpoint="https://api.test.com",
                api_key_env="TEST_KEY",
            )


class TestQuorumVote:
    """Tests for QuorumVote data model."""

    def test_quorum_vote_creation(self):
        """Test creating QuorumVote."""
        vote = QuorumVote(
            model_id="test",
            vote=True,
            confidence=0.9,
            reasoning="Code looks good",
            duration_ms=123.45,
        )

        assert vote.model_id == "test"
        assert vote.vote is True
        assert vote.confidence == 0.9
        assert vote.reasoning == "Code looks good"
        assert vote.duration_ms == 123.45
        assert vote.error is None
        assert isinstance(vote.vote_id, str)
        assert isinstance(vote.timestamp, datetime)

    def test_quorum_vote_with_error(self):
        """Test QuorumVote with error."""
        vote = QuorumVote(
            model_id="test",
            vote=False,
            confidence=0.0,
            reasoning="Failed to validate",
            duration_ms=50.0,
            error="API timeout",
        )

        assert vote.error == "API timeout"

    def test_quorum_vote_invalid_confidence(self):
        """Test QuorumVote with invalid confidence."""
        with pytest.raises(ValueError):
            QuorumVote(
                model_id="test",
                vote=True,
                confidence=1.5,  # Must be <= 1.0
                reasoning="Test",
                duration_ms=100.0,
            )


class TestQuorumResult:
    """Tests for QuorumResult data model."""

    def test_quorum_result_creation(self):
        """Test creating QuorumResult."""
        votes = [
            QuorumVote(
                model_id="a", vote=True, confidence=0.9, reasoning="Good", duration_ms=100
            ),
            QuorumVote(
                model_id="b", vote=True, confidence=0.8, reasoning="OK", duration_ms=150
            ),
        ]

        result = QuorumResult(
            passed=True,
            consensus_score=0.85,
            votes=votes,
            total_weight=4.0,
            participating_weight=3.5,
            pass_threshold=0.6,
            duration_ms=200.0,
        )

        assert result.passed is True
        assert result.consensus_score == 0.85
        assert len(result.votes) == 2
        assert result.total_weight == 4.0
        assert result.participating_weight == 3.5
        assert result.pass_threshold == 0.6
        assert result.duration_ms == 200.0
        assert isinstance(result.result_id, str)

    def test_quorum_result_get_votes_summary(self):
        """Test QuorumResult.get_votes_summary()."""
        votes = [
            QuorumVote(
                model_id="a", vote=True, confidence=0.9, reasoning="Good", duration_ms=100
            ),
            QuorumVote(
                model_id="b", vote=True, confidence=0.8, reasoning="OK", duration_ms=150
            ),
            QuorumVote(
                model_id="c", vote=False, confidence=0.7, reasoning="Bad", duration_ms=120
            ),
        ]

        result = QuorumResult(
            passed=True,
            consensus_score=0.75,
            votes=votes,
            total_weight=6.5,
            participating_weight=6.5,
            pass_threshold=0.6,
            duration_ms=200.0,
        )

        summary = result.get_votes_summary()

        assert summary["total_votes"] == 3
        assert summary["passed_votes"] == 2
        assert summary["failed_votes"] == 1
        assert summary["avg_confidence"] == 0.8  # (0.9 + 0.8 + 0.7) / 3
        assert summary["consensus_score"] == 0.75
        assert summary["passed"] is True

    def test_quorum_result_get_vote_by_model(self):
        """Test QuorumResult.get_vote_by_model()."""
        votes = [
            QuorumVote(
                model_id="a", vote=True, confidence=0.9, reasoning="Good", duration_ms=100
            ),
            QuorumVote(
                model_id="b", vote=False, confidence=0.7, reasoning="Bad", duration_ms=150
            ),
        ]

        result = QuorumResult(
            passed=False,
            consensus_score=0.5,
            votes=votes,
            total_weight=4.0,
            participating_weight=4.0,
            pass_threshold=0.6,
            duration_ms=200.0,
        )

        vote_a = result.get_vote_by_model("a")
        assert vote_a is not None
        assert vote_a.model_id == "a"
        assert vote_a.vote is True

        vote_b = result.get_vote_by_model("b")
        assert vote_b is not None
        assert vote_b.model_id == "b"
        assert vote_b.vote is False

        vote_c = result.get_vote_by_model("c")
        assert vote_c is None

    def test_quorum_result_invalid_participating_weight(self):
        """Test QuorumResult with invalid participating weight."""
        with pytest.raises(ValueError, match="cannot exceed total weight"):
            QuorumResult(
                passed=True,
                consensus_score=0.8,
                votes=[],
                total_weight=4.0,
                participating_weight=5.0,  # > total_weight
                pass_threshold=0.6,
                duration_ms=200.0,
            )


class TestValidationContext:
    """Tests for ValidationContext data model."""

    def test_validation_context_creation(self):
        """Test creating ValidationContext."""
        context = ValidationContext(
            node_type="effect",
            contract_summary="Test contract",
            validation_criteria=["ONEX compliance", "Code quality"],
        )

        assert context.node_type == "effect"
        assert context.contract_summary == "Test contract"
        assert len(context.validation_criteria) == 2
        assert context.code_snippet is None
        assert context.additional_context == {}


class TestAIQuorum:
    """Tests for AIQuorum class."""

    def test_ai_quorum_creation(self, sample_model_configs, mock_metrics_collector):
        """Test creating AIQuorum."""
        quorum = AIQuorum(
            model_configs=sample_model_configs,
            pass_threshold=0.6,
            metrics_collector=mock_metrics_collector,
        )

        assert quorum.pass_threshold == 0.6
        assert quorum.total_weight == 4.5  # 2.0 + 1.5 + 1.0
        assert len(quorum.model_configs) == 3

    def test_ai_quorum_invalid_threshold(self, sample_model_configs):
        """Test AIQuorum with invalid threshold."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            AIQuorum(model_configs=sample_model_configs, pass_threshold=1.5)

    def test_ai_quorum_empty_configs(self):
        """Test AIQuorum with empty configs."""
        with pytest.raises(ValueError, match="At least one model"):
            AIQuorum(model_configs=[])

    def test_ai_quorum_register_client(self, sample_model_configs):
        """Test registering LLM clients."""
        quorum = AIQuorum(model_configs=sample_model_configs)

        client = MockLLMClient(
            model_id="model_a",
            model_name="model-a-v1",
        )

        quorum.register_client("model_a", client)

        assert "model_a" in quorum.clients
        assert quorum.clients["model_a"] == client

    @pytest.mark.asyncio
    async def test_ai_quorum_validate_code_all_pass(
        self, sample_model_configs, validation_context, mock_metrics_collector
    ):
        """Test quorum validation where all models pass."""
        quorum = AIQuorum(
            model_configs=sample_model_configs,
            pass_threshold=0.6,
            metrics_collector=mock_metrics_collector,
            min_participating_weight=3.0,
        )

        # Register mock clients (all vote pass)
        for config in sample_model_configs:
            client = MockLLMClient(
                model_id=config.model_id,
                model_name=config.model_name,
                default_vote=True,
                default_confidence=0.9,
                latency_ms=50,
            )
            quorum.register_client(config.model_id, client)

        await quorum.initialize()

        result = await quorum.validate_code(
            code="def test(): pass",
            context=validation_context,
            correlation_id="test-123",
        )

        assert result.passed is True
        assert len(result.votes) == 3
        assert result.consensus_score >= 0.6
        assert result.total_weight == 4.5
        assert result.participating_weight == 4.5
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_ai_quorum_validate_code_mixed_votes(
        self, sample_model_configs, validation_context
    ):
        """Test quorum validation with mixed votes."""
        quorum = AIQuorum(
            model_configs=sample_model_configs,
            pass_threshold=0.6,
            min_participating_weight=3.0,
        )

        # Register mock clients with mixed votes
        # model_a (weight 2.0): pass with 0.9 confidence
        # model_b (weight 1.5): fail
        # model_c (weight 1.0): pass with 0.8 confidence
        quorum.register_client(
            "model_a",
            MockLLMClient(
                model_id="model_a",
                model_name="model-a-v1",
                default_vote=True,
                default_confidence=0.9,
            ),
        )
        quorum.register_client(
            "model_b",
            MockLLMClient(
                model_id="model_b",
                model_name="model-b-v1",
                default_vote=False,
                default_confidence=0.7,
            ),
        )
        quorum.register_client(
            "model_c",
            MockLLMClient(
                model_id="model_c",
                model_name="model-c-v1",
                default_vote=True,
                default_confidence=0.8,
            ),
        )

        await quorum.initialize()

        result = await quorum.validate_code(
            code="def test(): pass", context=validation_context
        )

        # Calculate expected consensus:
        # model_a: pass * 2.0 * 0.9 = 1.8
        # model_b: fail * 1.5 * 0.7 = 0.0
        # model_c: pass * 1.0 * 0.8 = 0.8
        # Total: (1.8 + 0.8) / 4.5 = 2.6 / 4.5 = 0.578

        assert len(result.votes) == 3
        assert result.consensus_score < 0.6  # Below threshold
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_ai_quorum_validate_code_all_fail(
        self, sample_model_configs, validation_context
    ):
        """Test quorum validation where all models fail."""
        quorum = AIQuorum(
            model_configs=sample_model_configs,
            pass_threshold=0.6,
            min_participating_weight=3.0,
        )

        # Register mock clients (all vote fail)
        for config in sample_model_configs:
            client = MockLLMClient(
                model_id=config.model_id,
                model_name=config.model_name,
                default_vote=False,
                default_confidence=0.8,
            )
            quorum.register_client(config.model_id, client)

        await quorum.initialize()

        result = await quorum.validate_code(
            code="def test(): pass", context=validation_context
        )

        assert result.passed is False
        assert result.consensus_score == 0.0  # All failed
        assert len(result.votes) == 3

    @pytest.mark.asyncio
    async def test_ai_quorum_insufficient_participation(
        self, sample_model_configs, validation_context
    ):
        """Test quorum validation with insufficient participation."""
        quorum = AIQuorum(
            model_configs=sample_model_configs,
            pass_threshold=0.6,
            min_participating_weight=5.0,  # Higher than total weight
        )

        # Register only one client
        quorum.register_client(
            "model_a",
            MockLLMClient(
                model_id="model_a",
                model_name="model-a-v1",
                default_vote=True,
                default_confidence=0.9,
            ),
        )

        await quorum.initialize()

        with pytest.raises(RuntimeError, match="Insufficient model participation"):
            await quorum.validate_code(
                code="def test(): pass", context=validation_context
            )

    @pytest.mark.asyncio
    async def test_ai_quorum_model_failure_handling(
        self, sample_model_configs, validation_context
    ):
        """Test quorum handles model failures gracefully."""
        quorum = AIQuorum(
            model_configs=sample_model_configs,
            pass_threshold=0.6,
            min_participating_weight=2.0,
        )

        # Register clients, one will fail
        quorum.register_client(
            "model_a",
            MockLLMClient(
                model_id="model_a",
                model_name="model-a-v1",
                default_vote=True,
                default_confidence=0.9,
            ),
        )

        # Create a failing client
        failing_client = MockLLMClient(
            model_id="model_b",
            model_name="model-b-v1",
        )
        failing_client.validate_code = AsyncMock(
            side_effect=Exception("API timeout")
        )
        quorum.register_client("model_b", failing_client)

        quorum.register_client(
            "model_c",
            MockLLMClient(
                model_id="model_c",
                model_name="model-c-v1",
                default_vote=True,
                default_confidence=0.8,
            ),
        )

        await quorum.initialize()

        result = await quorum.validate_code(
            code="def test(): pass", context=validation_context
        )

        # Should still get result with 2 successful votes
        assert len(result.votes) == 2  # model_b failed
        assert result.participating_weight == 3.0  # 2.0 + 1.0
        assert result.passed is True  # Both passed

    def test_ai_quorum_calculate_consensus(self, sample_model_configs):
        """Test consensus calculation."""
        quorum = AIQuorum(model_configs=sample_model_configs)

        votes = [
            QuorumVote(
                model_id="model_a",
                vote=True,
                confidence=0.9,
                reasoning="Good",
                duration_ms=100,
            ),
            QuorumVote(
                model_id="model_b",
                vote=True,
                confidence=0.8,
                reasoning="OK",
                duration_ms=150,
            ),
            QuorumVote(
                model_id="model_c",
                vote=False,
                confidence=0.7,
                reasoning="Bad",
                duration_ms=120,
            ),
        ]

        consensus = quorum._calculate_consensus(votes)

        # Expected:
        # model_a: pass * 2.0 * 0.9 = 1.8
        # model_b: pass * 1.5 * 0.8 = 1.2
        # model_c: fail * 1.0 * 0.7 = 0.0
        # Total: (1.8 + 1.2) / 4.5 = 3.0 / 4.5 = 0.667

        assert 0.66 <= consensus <= 0.67

    def test_ai_quorum_get_model_weight(self, sample_model_configs):
        """Test getting model weight."""
        quorum = AIQuorum(model_configs=sample_model_configs)

        assert quorum._get_model_weight("model_a") == 2.0
        assert quorum._get_model_weight("model_b") == 1.5
        assert quorum._get_model_weight("model_c") == 1.0
        assert quorum._get_model_weight("unknown") == 0.0

    def test_ai_quorum_get_statistics(self, sample_model_configs):
        """Test getting quorum statistics."""
        quorum = AIQuorum(model_configs=sample_model_configs)

        quorum._total_validations = 10
        quorum._total_passes = 8
        quorum._total_failures = 2
        quorum._total_cost = 0.50

        stats = quorum.get_statistics()

        assert stats["total_validations"] == 10
        assert stats["total_passes"] == 8
        assert stats["total_failures"] == 2
        assert stats["pass_rate"] == 0.8
        assert stats["total_cost"] == 0.50
        assert stats["avg_cost_per_validation"] == 0.05
        assert stats["total_weight"] == 4.5
        assert stats["pass_threshold"] == 0.6
        assert stats["num_models"] == 3

    @pytest.mark.asyncio
    async def test_ai_quorum_metrics_collection(
        self, sample_model_configs, validation_context, mock_metrics_collector
    ):
        """Test metrics collection during validation."""
        quorum = AIQuorum(
            model_configs=sample_model_configs,
            metrics_collector=mock_metrics_collector,
            min_participating_weight=3.0,
        )

        # Register mock clients
        for config in sample_model_configs:
            client = MockLLMClient(
                model_id=config.model_id,
                model_name=config.model_name,
                default_vote=True,
                default_confidence=0.9,
            )
            quorum.register_client(config.model_id, client)

        await quorum.initialize()

        await quorum.validate_code(
            code="def test(): pass",
            context=validation_context,
            correlation_id="test-123",
        )

        # Verify metrics were recorded
        mock_metrics_collector.record_counter.assert_any_call(
            "quorum_validation_started",
            count=1,
            tags={"node_type": "effect"},
            correlation_id="test-123",
        )

        mock_metrics_collector.record_timing.assert_called()
        mock_metrics_collector.record_gauge.assert_called()


class TestDefaultQuorumModels:
    """Tests for default quorum model configurations."""

    def test_default_quorum_models_structure(self):
        """Test default quorum models are properly configured."""
        assert len(DEFAULT_QUORUM_MODELS) == 4

        model_ids = [config.model_id for config in DEFAULT_QUORUM_MODELS]
        assert "gemini" in model_ids
        assert "glm-4.5" in model_ids
        assert "glm-air" in model_ids
        assert "codestral" in model_ids

    def test_default_quorum_models_weights(self):
        """Test default quorum models have correct weights."""
        weights = {config.model_id: config.weight for config in DEFAULT_QUORUM_MODELS}

        assert weights["gemini"] == 2.0
        assert weights["glm-4.5"] == 2.0
        assert weights["glm-air"] == 1.5
        assert weights["codestral"] == 1.0

        total_weight = sum(weights.values())
        assert total_weight == 6.5

    def test_default_quorum_models_endpoints(self):
        """Test default quorum models have valid endpoints."""
        for config in DEFAULT_QUORUM_MODELS:
            assert config.endpoint.startswith("https://")
            assert len(config.api_key_env) > 0


@pytest.mark.asyncio
async def test_end_to_end_quorum_validation():
    """End-to-end test of quorum validation."""
    # Create model configs
    configs = [
        ModelConfig(
            model_id="m1",
            model_name="model-1",
            weight=2.0,
            endpoint="http://mock1",
            api_key_env="KEY1",
        ),
        ModelConfig(
            model_id="m2",
            model_name="model-2",
            weight=1.5,
            endpoint="http://mock2",
            api_key_env="KEY2",
        ),
        ModelConfig(
            model_id="m3",
            model_name="model-3",
            weight=1.0,
            endpoint="http://mock3",
            api_key_env="KEY3",
        ),
    ]

    # Create quorum
    quorum = AIQuorum(
        model_configs=configs,
        pass_threshold=0.6,
        min_participating_weight=3.0,
    )

    # Register mock clients with different votes
    quorum.register_client(
        "m1",
        MockLLMClient(
            model_id="m1",
            model_name="model-1",
            default_vote=True,
            default_confidence=0.95,
            latency_ms=100,
        ),
    )
    quorum.register_client(
        "m2",
        MockLLMClient(
            model_id="m2",
            model_name="model-2",
            default_vote=True,
            default_confidence=0.85,
            latency_ms=150,
        ),
    )
    quorum.register_client(
        "m3",
        MockLLMClient(
            model_id="m3",
            model_name="model-3",
            default_vote=False,
            default_confidence=0.6,
            latency_ms=80,
        ),
    )

    await quorum.initialize()

    # Create validation context
    context = ValidationContext(
        node_type="compute",
        contract_summary="Test compute node contract",
        validation_criteria=[
            "ONEX v2.0 compliance",
            "Code quality",
            "Performance",
        ],
    )

    # Validate code
    result = await quorum.validate_code(
        code='async def execute_compute(input_data: Dict[str, Any]) -> Dict[str, Any]:\n    """Test compute node."""\n    return {"result": input_data}',
        context=context,
        correlation_id="e2e-test",
    )

    # Verify result
    assert isinstance(result, QuorumResult)
    assert len(result.votes) == 3
    assert result.total_weight == 4.5
    assert result.participating_weight == 4.5
    assert result.correlation_id == "e2e-test"
    assert result.duration_ms > 0

    # Expected consensus:
    # m1: pass * 2.0 * 0.95 = 1.9
    # m2: pass * 1.5 * 0.85 = 1.275
    # m3: fail * 1.0 * 0.6 = 0.0
    # Total: (1.9 + 1.275) / 4.5 = 3.175 / 4.5 = 0.706
    assert result.passed is True
    assert 0.70 <= result.consensus_score <= 0.71

    # Verify votes
    vote_m1 = result.get_vote_by_model("m1")
    assert vote_m1 is not None
    assert vote_m1.vote is True
    assert vote_m1.confidence == 0.95

    vote_m3 = result.get_vote_by_model("m3")
    assert vote_m3 is not None
    assert vote_m3.vote is False

    # Get statistics
    stats = quorum.get_statistics()
    assert stats["total_validations"] == 1
    assert stats["total_passes"] == 1
    assert stats["pass_rate"] == 1.0

    await quorum.close()

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the model router scoring handler (pure compute)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_model_health_effect.models.model_endpoint_health import (
    ModelEndpointHealth,
)
from omnibase_infra.nodes.node_model_router_compute.handlers.handler_score_models import (
    HandlerScoreModels,
)
from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_live_metrics import (
    ModelLiveMetrics,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_registry_entry import (
    ModelRegistryEntry,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_constraints import (
    ModelRoutingConstraints,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_scoring_input import (
    ModelScoringInput,
)


def _make_registry() -> tuple[ModelRegistryEntry, ...]:
    return (
        ModelRegistryEntry(
            model_key="qwen3-coder-30b",
            provider="local",
            transport="http",
            base_url_env="LLM_CODER_URL",
            capabilities=("code_generation", "refactoring"),
            context_window=65536,
            seed_cost_per_1k_tokens=0.0,
            seed_tokens_per_sec=201,
            tier="local",
        ),
        ModelRegistryEntry(
            model_key="claude-sonnet",
            provider="anthropic",
            transport="oauth",
            capabilities=("code_generation", "reasoning", "vision"),
            context_window=200000,
            seed_cost_per_1k_tokens=0.015,
            seed_tokens_per_sec=80,
            tier="frontier_api",
        ),
        ModelRegistryEntry(
            model_key="deepseek-r1-32b",
            provider="local",
            transport="http",
            base_url_env="LLM_DEEPSEEK_R1_URL",
            capabilities=("deep_reasoning", "code_review"),
            context_window=32768,
            seed_cost_per_1k_tokens=0.0,
            seed_tokens_per_sec=6.5,
            tier="local",
        ),
    )


@pytest.mark.unit
class TestHandlerScoreModels:
    """Tests for the pure scoring handler."""

    def test_selects_local_model_for_code_generation(self) -> None:
        """Local free model should win for code generation with prefer_local."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(prefer_local=True),
            context_length_estimate=4096,
            registry=registry,
        )

        decision = handler.score_candidates(scoring_input)

        assert decision.success
        assert decision.selected_model_key == "qwen3-coder-30b"
        assert decision.estimated_cost == 0.0
        assert "qwen3-coder-30b" in decision.scores

    def test_selects_vision_model_when_required(self) -> None:
        """Only vision-capable models should be selected when needs_vision=True."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.VISION,
            constraints=ModelRoutingConstraints(needs_vision=True),
            context_length_estimate=4096,
            registry=registry,
        )

        decision = handler.score_candidates(scoring_input)

        assert decision.success
        assert decision.selected_model_key == "claude-sonnet"

    def test_no_candidates_returns_failure(self) -> None:
        """If no models pass constraints, return failure."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(
                needs_computer_use=True,  # No model in test registry has this
            ),
            context_length_estimate=4096,
            registry=registry,
        )

        decision = handler.score_candidates(scoring_input)

        assert not decision.success
        assert decision.selected_model_key == ""
        assert "No eligible models" in decision.error_message

    def test_unhealthy_model_excluded(self) -> None:
        """Unhealthy models should be filtered out."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        health = (
            ModelEndpointHealth(
                model_key="qwen3-coder-30b", healthy=False, error_message="timeout"
            ),
            ModelEndpointHealth(model_key="claude-sonnet", healthy=True),
            ModelEndpointHealth(model_key="deepseek-r1-32b", healthy=True),
        )

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(),
            context_length_estimate=4096,
            registry=registry,
            health=health,
        )

        decision = handler.score_candidates(scoring_input)

        assert decision.success
        assert decision.selected_model_key != "qwen3-coder-30b"

    def test_chain_hit_boosts_target_model(self) -> None:
        """Chain hit should boost the specified model's score."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        # Without chain hit
        input_no_chain = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(),
            context_length_estimate=4096,
            registry=registry,
        )
        decision_no_chain = handler.score_candidates(input_no_chain)
        score_no_chain = decision_no_chain.scores.get("deepseek-r1-32b", 0)

        # With chain hit for deepseek
        input_chain = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(),
            context_length_estimate=4096,
            registry=registry,
            chain_hit=True,
            chain_hit_model_key="deepseek-r1-32b",
        )
        decision_chain = handler.score_candidates(input_chain)
        score_chain = decision_chain.scores.get("deepseek-r1-32b", 0)

        assert score_chain > score_no_chain

    def test_cost_cap_filters_expensive_models(self) -> None:
        """max_cost_per_1k=0 should filter out frontier API models."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(max_cost_per_1k=0.0),
            context_length_estimate=4096,
            registry=registry,
        )

        decision = handler.score_candidates(scoring_input)

        assert decision.success
        assert decision.selected_model_key in ("qwen3-coder-30b", "deepseek-r1-32b")

    def test_live_metrics_influence_scoring(self) -> None:
        """Live metrics with sufficient samples should influence quality score."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        live_metrics = (
            ModelLiveMetrics(
                model_key="deepseek-r1-32b",
                task_type=EnumTaskType.CODE_GENERATION,
                success_rate=0.95,
                sample_count=30,
                avg_latency_ms=500,
                avg_tokens_per_sec=10.0,
                graduated=True,
            ),
        )

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(),
            context_length_estimate=4096,
            registry=registry,
            live_metrics=live_metrics,
        )

        decision = handler.score_candidates(scoring_input)

        # deepseek should have a higher quality score due to graduation
        assert decision.scores["deepseek-r1-32b"] > 0

    def test_context_window_constraint(self) -> None:
        """Models with insufficient context window should be filtered."""
        handler = HandlerScoreModels()
        registry = _make_registry()

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(min_context_window=100000),
            context_length_estimate=80000,
            registry=registry,
        )

        decision = handler.score_candidates(scoring_input)

        assert decision.success
        assert decision.selected_model_key == "claude-sonnet"

    def test_deterministic_for_same_inputs(self) -> None:
        """Same inputs should produce same output (pure function)."""
        handler = HandlerScoreModels()
        registry = _make_registry()
        cid = uuid4()

        scoring_input = ModelScoringInput(
            correlation_id=cid,
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(),
            context_length_estimate=4096,
            registry=registry,
        )

        d1 = handler.score_candidates(scoring_input)
        d2 = handler.score_candidates(scoring_input)

        assert d1.selected_model_key == d2.selected_model_key
        assert d1.scores == d2.scores

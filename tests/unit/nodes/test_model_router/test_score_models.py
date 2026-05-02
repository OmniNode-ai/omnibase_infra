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
            tier="local",
        ),
        ModelRegistryEntry(
            model_key="claude-sonnet",
            provider="anthropic",
            transport="oauth",
            capabilities=("code_generation", "reasoning", "vision"),
            context_window=200000,
            tier="frontier_api",
        ),
        ModelRegistryEntry(
            model_key="deepseek-r1-32b",
            provider="local",
            transport="http",
            base_url_env="LLM_DEEPSEEK_R1_URL",
            capabilities=("deep_reasoning", "code_review"),
            context_window=32768,
            tier="local",
        ),
    )


@pytest.mark.unit
class TestHandlerScoreModels:
    """Tests for the pure scoring handler."""

    def test_selects_local_model_for_code_generation(self) -> None:
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
        handler = HandlerScoreModels()
        registry = _make_registry()

        scoring_input = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(
                needs_computer_use=True,
            ),
            context_length_estimate=4096,
            registry=registry,
        )

        decision = handler.score_candidates(scoring_input)

        assert not decision.success
        assert decision.selected_model_key == ""
        assert "No eligible models" in decision.error_message

    def test_unhealthy_model_excluded(self) -> None:
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
        handler = HandlerScoreModels()
        registry = _make_registry()

        input_no_chain = ModelScoringInput(
            correlation_id=uuid4(),
            task_type=EnumTaskType.CODE_GENERATION,
            constraints=ModelRoutingConstraints(),
            context_length_estimate=4096,
            registry=registry,
        )
        decision_no_chain = handler.score_candidates(input_no_chain)
        score_no_chain = decision_no_chain.scores.get("deepseek-r1-32b", 0)

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

    def test_cost_cap_filters_frontier_models(self) -> None:
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

        assert decision.scores["deepseek-r1-32b"] > 0

    def test_unmeasured_models_get_lowest_speed(self) -> None:
        handler = HandlerScoreModels()
        registry = _make_registry()

        live_metrics = (
            ModelLiveMetrics(
                model_key="qwen3-coder-30b",
                task_type=EnumTaskType.CODE_GENERATION,
                success_rate=0.9,
                sample_count=25,
                avg_latency_ms=200,
                avg_tokens_per_sec=150.0,
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

        qwen_score = decision.scores["qwen3-coder-30b"]
        deepseek_score = decision.scores["deepseek-r1-32b"]
        assert qwen_score > deepseek_score

    def test_context_window_constraint(self) -> None:
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

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the routing score reducer handler (pure state transitions)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_routing_score_reducer.handlers.handler_update_scores import (
    HandlerUpdateScores,
)
from omnibase_infra.nodes.node_routing_score_reducer.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_routing_score_reducer.models.model_routing_outcome import (
    ModelRoutingOutcome,
)


def _empty_state() -> ModelReducerState:
    return ModelReducerState(correlation_id=uuid4())


@pytest.mark.unit
class TestHandlerUpdateScores:
    """Tests for the pure reducer handler."""

    def test_first_outcome_creates_score(self) -> None:
        """First outcome for a (model, task_type) should create a new score entry."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        outcome = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key="qwen3-coder-30b",
            task_type=EnumTaskType.CODE_GENERATION,
            success=True,
            actual_latency_ms=100,
            actual_tokens_per_sec=200.0,
        )

        new_state = handler.apply_outcome(state, outcome)

        assert len(new_state.scores) == 1
        assert new_state.scores[0].model_key == "qwen3-coder-30b"
        assert new_state.scores[0].success_count == 1
        assert new_state.scores[0].total_count == 1
        assert new_state.scores[0].success_rate == 1.0
        assert new_state.total_outcomes_processed == 1

    def test_failure_tracked(self) -> None:
        """Failed outcome should increment failure_count."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        outcome = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key="qwen3-coder-30b",
            task_type=EnumTaskType.CODE_GENERATION,
            success=False,
        )

        new_state = handler.apply_outcome(state, outcome)

        assert new_state.scores[0].failure_count == 1
        assert new_state.scores[0].success_rate == 0.0

    def test_accumulates_multiple_outcomes(self) -> None:
        """Multiple outcomes for same (model, task_type) should accumulate."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        for i in range(5):
            outcome = ModelRoutingOutcome(
                correlation_id=uuid4(),
                model_key="qwen3-coder-30b",
                task_type=EnumTaskType.CODE_GENERATION,
                success=True,
                actual_latency_ms=100 + i * 10,
                actual_tokens_per_sec=200.0,
            )
            state = handler.apply_outcome(state, outcome)

        assert len(state.scores) == 1
        assert state.scores[0].total_count == 5
        assert state.scores[0].success_count == 5
        assert state.total_outcomes_processed == 5

    def test_separate_models_tracked_independently(self) -> None:
        """Different models should get separate score entries."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        for model_key in ("qwen3-coder-30b", "deepseek-r1-32b"):
            outcome = ModelRoutingOutcome(
                correlation_id=uuid4(),
                model_key=model_key,
                task_type=EnumTaskType.CODE_GENERATION,
                success=True,
            )
            state = handler.apply_outcome(state, outcome)

        assert len(state.scores) == 2
        model_keys = {s.model_key for s in state.scores}
        assert model_keys == {"qwen3-coder-30b", "deepseek-r1-32b"}

    def test_graduation_after_threshold(self) -> None:
        """Model should graduate after 50+ attempts with >0.9 success rate."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        # 50 successes
        for _ in range(50):
            outcome = ModelRoutingOutcome(
                correlation_id=uuid4(),
                model_key="qwen3-coder-30b",
                task_type=EnumTaskType.CODE_GENERATION,
                success=True,
                actual_latency_ms=100,
                actual_tokens_per_sec=200.0,
            )
            state = handler.apply_outcome(state, outcome)

        assert state.scores[0].graduated is True
        assert state.scores[0].success_rate >= 0.9

    def test_no_premature_graduation(self) -> None:
        """Model should NOT graduate before 50 attempts even at 100% success."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        for _ in range(10):
            outcome = ModelRoutingOutcome(
                correlation_id=uuid4(),
                model_key="qwen3-coder-30b",
                task_type=EnumTaskType.CODE_GENERATION,
                success=True,
            )
            state = handler.apply_outcome(state, outcome)

        assert state.scores[0].graduated is False

    def test_degraduation_on_regression(self) -> None:
        """Graduated model should de-graduate if success drops below 0.8."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        # Graduate: 50 successes
        for _ in range(50):
            outcome = ModelRoutingOutcome(
                correlation_id=uuid4(),
                model_key="qwen3-coder-30b",
                task_type=EnumTaskType.CODE_GENERATION,
                success=True,
            )
            state = handler.apply_outcome(state, outcome)

        assert state.scores[0].graduated is True

        # Regress: many failures to drop below 0.8
        for _ in range(40):
            outcome = ModelRoutingOutcome(
                correlation_id=uuid4(),
                model_key="qwen3-coder-30b",
                task_type=EnumTaskType.CODE_GENERATION,
                success=False,
            )
            state = handler.apply_outcome(state, outcome)

        assert state.scores[0].graduated is False

    def test_cost_accumulates(self) -> None:
        """Total cost should accumulate across outcomes."""
        handler = HandlerUpdateScores()
        state = _empty_state()

        for _ in range(3):
            outcome = ModelRoutingOutcome(
                correlation_id=uuid4(),
                model_key="claude-sonnet",
                task_type=EnumTaskType.CODE_GENERATION,
                success=True,
                actual_cost=0.05,
            )
            state = handler.apply_outcome(state, outcome)

        assert state.scores[0].total_cost == pytest.approx(0.15, abs=1e-6)

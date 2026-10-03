# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the routing score reducer handler (pure state transitions)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_routing_score_reducer.handlers.handler_update_scores import (
    HandlerUpdateScores,
)
from omnibase_infra.nodes.node_routing_score_reducer.models.model_capability_score import (
    ModelCapabilityScore,
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

    def test_routing_score_handler_classification(self) -> None:
        """Reducer handlers expose the node-handler and compute classifications."""
        handler = HandlerUpdateScores()

        assert handler.handler_type is EnumHandlerType.NODE_HANDLER
        assert handler.handler_category is EnumHandlerTypeCategory.COMPUTE

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

    @pytest.mark.parametrize("success", [True, False])
    def test_routing_score_unknown_backend_preserves_existing_scores(
        self, success: bool
    ) -> None:
        """An unseen backend appends its score without changing another backend."""
        existing = ModelCapabilityScore(
            model_key="known-backend",
            task_type=EnumTaskType.CODE_GENERATION,
            success_count=3,
            total_count=3,
            success_rate=1.0,
        )
        state = ModelReducerState(
            correlation_id=uuid4(), scores=(existing,), total_outcomes_processed=3
        )
        before = state.model_dump()
        outcome = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key="unknown-backend",
            task_type=EnumTaskType.CODE_GENERATION,
            success=success,
            actual_latency_ms=123,
            actual_tokens_per_sec=45.5,
            actual_cost=0.125,
        )

        updated = HandlerUpdateScores().apply_outcome(state, outcome)

        assert state.model_dump() == before
        assert updated.scores[0] is existing
        score = updated.scores[1]
        assert score.model_key == outcome.model_key
        assert score.task_type == outcome.task_type
        assert (score.success_count, score.failure_count, score.total_count) == (
            int(success),
            int(not success),
            1,
        )
        assert score.success_rate == float(success)
        assert score.avg_latency_ms == 123
        assert score.avg_tokens_per_sec == 45.5
        assert score.total_cost == 0.125
        assert score.graduated is False
        assert updated.correlation_id == outcome.correlation_id
        assert updated.total_outcomes_processed == 4

    @pytest.mark.parametrize("success", [True, False])
    def test_routing_score_zero_attempt_backend_uses_first_observation(
        self, success: bool
    ) -> None:
        """A seeded backend with no attempts replaces its placeholder averages."""
        existing = ModelCapabilityScore(
            model_key="seeded-backend",
            task_type=EnumTaskType.CODE_GENERATION,
            avg_latency_ms=999,
            avg_tokens_per_sec=999.0,
        )
        state = ModelReducerState(correlation_id=uuid4(), scores=(existing,))
        before = state.model_dump()
        outcome = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key=existing.model_key,
            task_type=existing.task_type,
            success=success,
            actual_latency_ms=120,
            actual_tokens_per_sec=60.0,
            actual_cost=0.25,
        )

        updated = HandlerUpdateScores().apply_outcome(state, outcome)

        assert state.model_dump() == before
        assert len(updated.scores) == 1
        score = updated.scores[0]
        assert (score.success_count, score.failure_count, score.total_count) == (
            int(success),
            int(not success),
            1,
        )
        assert score.success_rate == float(success)
        assert score.avg_latency_ms == 120
        assert score.avg_tokens_per_sec == 60.0
        assert score.total_cost == 0.25
        assert score.graduated is False
        assert updated.correlation_id == outcome.correlation_id
        assert updated.total_outcomes_processed == 1

    @pytest.mark.parametrize(
        ("total_count", "success_count", "failure_count", "success", "expected"),
        [
            pytest.param(
                99, 99, 0, True, (100, 0, 100, 1.0, 101, 51.0), id="at-ceiling"
            ),
            pytest.param(
                100, 100, 0, True, (100, 0, 100, 1.0, 101, 51.0), id="above-ceiling"
            ),
            pytest.param(
                100, 90, 10, False, (89, 10, 99, 0.899, 101, 51.01), id="mixed-window"
            ),
            # The score model accepts sparse counters; proportional truncation
            # drops the new observation to zero, so the denominator needs a floor.
            pytest.param(
                100, 0, 0, True, (0, 0, 1, 0.0, 200, 150.0), id="denominator-floor"
            ),
        ],
    )
    def test_routing_score_rolling_window_clamps(
        self,
        total_count: int,
        success_count: int,
        failure_count: int,
        success: bool,
        expected: tuple[int, int, int, float, int, float],
    ) -> None:
        """The rolling window caps counts and keeps a nonzero denominator."""
        existing = ModelCapabilityScore(
            model_key="window-backend",
            task_type=EnumTaskType.CODE_GENERATION,
            total_count=total_count,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_count / total_count,
            avg_latency_ms=100,
            avg_tokens_per_sec=50.0,
            total_cost=0.1,
        )
        state = ModelReducerState(correlation_id=uuid4(), scores=(existing,))
        before = state.model_dump()
        outcome = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key=existing.model_key,
            task_type=existing.task_type,
            success=success,
            actual_latency_ms=200,
            actual_tokens_per_sec=150.0,
            actual_cost=0.2,
        )

        updated = HandlerUpdateScores().apply_outcome(state, outcome)

        assert state.model_dump() == before
        score = updated.scores[0]
        assert (
            score.success_count,
            score.failure_count,
            score.total_count,
            score.success_rate,
            score.avg_latency_ms,
            score.avg_tokens_per_sec,
        ) == expected
        assert score.total_cost == pytest.approx(0.3)

    def test_routing_score_update_preserves_other_task_and_backend(self) -> None:
        """Only the matching backend and task pair is replaced in place."""
        other_task = ModelCapabilityScore(
            model_key="same-backend", task_type=EnumTaskType.REASONING
        )
        target = ModelCapabilityScore(
            model_key="same-backend", task_type=EnumTaskType.CODE_GENERATION
        )
        other_backend = ModelCapabilityScore(
            model_key="other-backend", task_type=EnumTaskType.CODE_GENERATION
        )
        state = ModelReducerState(
            correlation_id=uuid4(), scores=(other_task, target, other_backend)
        )
        outcome = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key=target.model_key,
            task_type=target.task_type,
            success=True,
        )

        updated = HandlerUpdateScores().apply_outcome(state, outcome)

        assert len(updated.scores) == 3
        assert updated.scores[0] is other_task
        assert updated.scores[1].success_count == 1
        assert updated.scores[2] is other_backend
        assert target.total_count == 0

    @pytest.mark.parametrize(
        ("success_count", "graduated", "success", "expected_rate"),
        [
            pytest.param(39, True, True, 0.8, id="degraduation-boundary"),
            pytest.param(44, True, False, 0.88, id="graduated-hysteresis"),
            pytest.param(39, False, False, 0.78, id="below-graduation"),
        ],
    )
    def test_routing_score_graduation_state_is_retained(
        self,
        success_count: int,
        graduated: bool,
        success: bool,
        expected_rate: float,
    ) -> None:
        """Graduation is retained at its lower boundary and below the upper one."""
        existing = ModelCapabilityScore(
            model_key="threshold-backend",
            task_type=EnumTaskType.CODE_GENERATION,
            success_count=success_count,
            failure_count=49 - success_count,
            total_count=49,
            graduated=graduated,
        )
        state = ModelReducerState(correlation_id=uuid4(), scores=(existing,))
        outcome = ModelRoutingOutcome(
            correlation_id=uuid4(),
            model_key=existing.model_key,
            task_type=existing.task_type,
            success=success,
        )

        updated = HandlerUpdateScores().apply_outcome(state, outcome)

        assert updated.scores[0].total_count == 50
        assert updated.scores[0].success_rate == expected_rate
        assert updated.scores[0].graduated is graduated

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reducer handler — pure state transition for capability scores.

delta(state, event) -> new_state. No I/O.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_routing_score_reducer.models.model_capability_score import (
    ModelCapabilityScore,
)
from omnibase_infra.nodes.node_routing_score_reducer.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_routing_score_reducer.models.model_routing_outcome import (
    ModelRoutingOutcome,
)

logger = logging.getLogger(__name__)

# Graduation thresholds
_GRADUATION_MIN_ATTEMPTS = 50
_GRADUATION_SUCCESS_RATE = 0.9
_DEGRADUATION_SUCCESS_RATE = 0.8

# Rolling window size
_ROLLING_WINDOW = 100


def _find_score(
    scores: tuple[ModelCapabilityScore, ...],
    model_key: str,
    task_type: str,
) -> ModelCapabilityScore | None:
    """Find existing score for (model_key, task_type)."""
    for s in scores:
        if s.model_key == model_key and s.task_type.value == task_type:
            return s
    return None


def _update_rolling_avg(current: float, new_value: float, count: int) -> float:
    """Update a rolling average with a new data point."""
    if count <= 1:
        return new_value
    return current + (new_value - current) / count


class HandlerUpdateScores:
    """Pure reducer handler — updates capability scores from routing outcomes."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE  # Reducers are pure

    def apply_outcome(
        self,
        state: ModelReducerState,
        outcome: ModelRoutingOutcome,
    ) -> ModelReducerState:
        """Apply a routing outcome to the reducer state.

        Args:
            state: Current reducer state.
            outcome: New routing outcome event.

        Returns:
            New reducer state with updated scores.
        """
        existing = _find_score(state.scores, outcome.model_key, outcome.task_type.value)

        now = datetime.now(tz=UTC)

        if existing is None:
            # New (model, task_type) pair
            new_score = ModelCapabilityScore(
                model_key=outcome.model_key,
                task_type=outcome.task_type,
                success_count=1 if outcome.success else 0,
                failure_count=0 if outcome.success else 1,
                total_count=1,
                success_rate=1.0 if outcome.success else 0.0,
                avg_latency_ms=outcome.actual_latency_ms,
                avg_tokens_per_sec=outcome.actual_tokens_per_sec,
                total_cost=outcome.actual_cost,
                graduated=False,
                last_updated=now,
            )
            updated_scores = (*state.scores, new_score)
        else:
            # Update existing — apply rolling window cap
            total = min(existing.total_count + 1, _ROLLING_WINDOW)
            sc = existing.success_count + (1 if outcome.success else 0)
            fc = existing.failure_count + (0 if outcome.success else 1)

            # Cap to rolling window
            if existing.total_count >= _ROLLING_WINDOW:
                # Approximate: subtract proportionally
                ratio = _ROLLING_WINDOW / (existing.total_count + 1)
                sc = int(sc * ratio)
                fc = int(fc * ratio)
                total = sc + fc if (sc + fc) > 0 else 1

            success_rate = sc / total if total > 0 else 0.0

            # Graduation logic
            graduated = existing.graduated
            if total >= _GRADUATION_MIN_ATTEMPTS:
                if success_rate >= _GRADUATION_SUCCESS_RATE:
                    if not graduated:
                        logger.info(
                            "Model %s graduated for %s (rate=%.2f, n=%d)",
                            outcome.model_key,
                            outcome.task_type.value,
                            success_rate,
                            total,
                        )
                    graduated = True
                elif graduated and success_rate < _DEGRADUATION_SUCCESS_RATE:
                    logger.info(
                        "Model %s de-graduated for %s (rate=%.2f, n=%d)",
                        outcome.model_key,
                        outcome.task_type.value,
                        success_rate,
                        total,
                    )
                    graduated = False

            avg_latency = _update_rolling_avg(
                existing.avg_latency_ms, outcome.actual_latency_ms, total
            )
            avg_tps = _update_rolling_avg(
                existing.avg_tokens_per_sec, outcome.actual_tokens_per_sec, total
            )

            new_score = ModelCapabilityScore(
                model_key=outcome.model_key,
                task_type=outcome.task_type,
                success_count=sc,
                failure_count=fc,
                total_count=total,
                success_rate=round(success_rate, 4),
                avg_latency_ms=int(avg_latency),
                avg_tokens_per_sec=round(avg_tps, 2),
                total_cost=round(existing.total_cost + outcome.actual_cost, 6),
                graduated=graduated,
                last_updated=now,
            )

            updated_scores = tuple(
                new_score
                if (
                    s.model_key == outcome.model_key
                    and s.task_type == outcome.task_type
                )
                else s
                for s in state.scores
            )

        return ModelReducerState(
            correlation_id=outcome.correlation_id,
            scores=updated_scores,
            total_outcomes_processed=state.total_outcomes_processed + 1,
        )

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Cost delta between baseline and candidate runs.

Computes the difference in cost metrics to quantify the token/time/retry
savings (or overhead) from applying a pattern.  Negative deltas indicate
the candidate (with pattern) used *fewer* resources than the baseline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.baseline.model_cost_metrics import ModelCostMetrics


class ModelCostDelta(BaseModel):
    """Delta between baseline and candidate cost metrics.

    All deltas are computed as ``baseline - candidate``, so positive
    values indicate savings from the pattern, negative values indicate
    the pattern added overhead.

    Attributes:
        token_delta: Difference in total tokens (baseline - candidate).
        prompt_token_delta: Difference in prompt tokens.
        completion_token_delta: Difference in completion tokens.
        wall_time_delta_ms: Difference in wall-clock time (ms).
        retry_delta: Difference in retry count.
        token_savings_pct: Token savings as a percentage of baseline.
        time_savings_pct: Time savings as a percentage of baseline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    token_delta: int = Field(
        default=0,
        description="Difference in total tokens (baseline - candidate).",
    )
    prompt_token_delta: int = Field(
        default=0,
        description="Difference in prompt tokens (baseline - candidate).",
    )
    completion_token_delta: int = Field(
        default=0,
        description="Difference in completion tokens (baseline - candidate).",
    )
    wall_time_delta_ms: float = Field(
        default=0.0,
        description="Difference in wall-clock time in ms (baseline - candidate).",
    )
    retry_delta: int = Field(
        default=0,
        description="Difference in retry count (baseline - candidate).",
    )
    token_savings_pct: float = Field(
        default=0.0,
        description="Token savings as a percentage of baseline.",
    )
    time_savings_pct: float = Field(
        default=0.0,
        description="Time savings as a percentage of baseline.",
    )

    @staticmethod
    def from_metrics(
        baseline: ModelCostMetrics,
        candidate: ModelCostMetrics,
    ) -> ModelCostDelta:
        """Compute the cost delta between baseline and candidate.

        Args:
            baseline: Cost metrics from the baseline run.
            candidate: Cost metrics from the candidate run.

        Returns:
            A ``ModelCostDelta`` with all deltas and percentages computed.
        """
        token_delta = baseline.total_tokens - candidate.total_tokens
        prompt_delta = baseline.prompt_tokens - candidate.prompt_tokens
        completion_delta = baseline.completion_tokens - candidate.completion_tokens
        time_delta = baseline.wall_time_ms - candidate.wall_time_ms
        retry_delta = baseline.retry_count - candidate.retry_count

        token_pct = (
            (token_delta / baseline.total_tokens * 100.0)
            if baseline.total_tokens > 0
            else 0.0
        )
        time_pct = (
            (time_delta / baseline.wall_time_ms * 100.0)
            if baseline.wall_time_ms > 0.0
            else 0.0
        )

        return ModelCostDelta(
            token_delta=token_delta,
            prompt_token_delta=prompt_delta,
            completion_token_delta=completion_delta,
            wall_time_delta_ms=time_delta,
            retry_delta=retry_delta,
            token_savings_pct=round(token_pct, 2),
            time_savings_pct=round(time_pct, 2),
        )


__all__: list[str] = ["ModelCostDelta"]

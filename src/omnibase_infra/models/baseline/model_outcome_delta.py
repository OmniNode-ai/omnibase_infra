# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Outcome delta between baseline and candidate runs.

Computes the difference in outcome metrics to quantify quality
improvements from applying a pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.baseline.model_outcome_metrics import ModelOutcomeMetrics


class ModelOutcomeDelta(BaseModel):
    """Delta between baseline and candidate outcome metrics.

    Positive ``check_delta`` means the candidate passed more checks.
    Negative ``flake_rate_delta`` means the candidate had fewer flakes.

    Attributes:
        baseline_passed: Whether the baseline run passed.
        candidate_passed: Whether the candidate run passed.
        check_delta: Difference in passed checks (candidate - baseline).
        flake_rate_delta: Difference in flake rate (baseline - candidate).
        review_iteration_delta: Difference in review iterations
            (baseline - candidate).
        quality_improved: True if the candidate outcome is strictly
            better than the baseline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    baseline_passed: bool = Field(
        ...,
        description="Whether the baseline run passed.",
    )
    candidate_passed: bool = Field(
        ...,
        description="Whether the candidate run passed.",
    )
    check_delta: int = Field(
        default=0,
        description="Difference in passed checks (candidate - baseline).",
    )
    flake_rate_delta: float = Field(
        default=0.0,
        description="Difference in flake rate (baseline - candidate).",
    )
    review_iteration_delta: int = Field(
        default=0,
        description=("Difference in review iterations (baseline - candidate)."),
    )
    quality_improved: bool = Field(
        default=False,
        description=("True if the candidate outcome is strictly better than baseline."),
    )

    @staticmethod
    def from_metrics(
        baseline: ModelOutcomeMetrics,
        candidate: ModelOutcomeMetrics,
    ) -> ModelOutcomeDelta:
        """Compute the outcome delta between baseline and candidate.

        Quality is considered "improved" when:
        - Candidate passed and baseline did not, OR
        - Both passed but candidate has more passed checks, OR
        - Both passed, same checks, but candidate has lower flake rate.

        Args:
            baseline: Outcome metrics from the baseline run.
            candidate: Outcome metrics from the candidate run.

        Returns:
            A ``ModelOutcomeDelta`` with all deltas computed.
        """
        check_delta = candidate.passed_checks - baseline.passed_checks
        flake_delta = baseline.flake_rate - candidate.flake_rate
        review_delta = baseline.review_iterations - candidate.review_iterations

        quality_improved = False
        if candidate.passed and not baseline.passed:
            quality_improved = True
        elif candidate.passed and baseline.passed:
            if check_delta > 0 or (check_delta == 0 and flake_delta > 0):
                quality_improved = True

        return ModelOutcomeDelta(
            baseline_passed=baseline.passed,
            candidate_passed=candidate.passed,
            check_delta=check_delta,
            flake_rate_delta=round(flake_delta, 4),
            review_iteration_delta=review_delta,
            quality_improved=quality_improved,
        )


__all__: list[str] = ["ModelOutcomeDelta"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Attribution record for A/B baseline comparison results.

Stores the complete comparison between a baseline run and a candidate
run, including cost and outcome deltas.  Used for promotion decisions
at Tier 2+ to prove pattern ROI.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumLifecycleTier
from omnibase_infra.models.baseline.model_baseline_run_result import (
    ModelBaselineRunResult,
)
from omnibase_infra.models.baseline.model_cost_delta import ModelCostDelta
from omnibase_infra.models.baseline.model_outcome_delta import ModelOutcomeDelta


class ModelAttributionRecord(BaseModel):
    """Attribution record proving pattern ROI through A/B comparison.

    Ties together the baseline run, candidate run, and their deltas
    to form a complete evidence record for promotion decisions.

    Attributes:
        record_id: Unique identifier for this attribution record.
        pattern_id: Identifier of the pattern being evaluated.
        scenario_id: Identifier of the test scenario used.
        correlation_id: Correlation ID for distributed tracing.
        current_tier: Tier at which the comparison was performed.
        target_tier: Tier the pattern is being promoted to.
        baseline_result: Full result of the baseline run.
        candidate_result: Full result of the candidate run.
        cost_metrics: Cost delta between baseline and candidate.
        outcome_metrics: Outcome delta between baseline and candidate.
        roi_positive: Whether the pattern demonstrates positive ROI
            (cost savings AND quality maintained or improved).
        created_at: Timestamp when this record was created.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    record_id: UUID = Field(
        ...,
        description="Unique identifier for this attribution record.",
    )
    pattern_id: UUID = Field(
        ...,
        description="Identifier of the pattern being evaluated.",
    )
    scenario_id: UUID = Field(
        ...,
        description="Identifier of the test scenario used.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    current_tier: EnumLifecycleTier = Field(
        ...,
        description="Tier at which the comparison was performed.",
    )
    target_tier: EnumLifecycleTier = Field(
        ...,
        description="Tier the pattern is being promoted to.",
    )
    baseline_result: ModelBaselineRunResult = Field(
        ...,
        description="Full result of the baseline run.",
    )
    candidate_result: ModelBaselineRunResult = Field(
        ...,
        description="Full result of the candidate run.",
    )
    cost_metrics: ModelCostDelta = Field(
        ...,
        description="Cost delta between baseline and candidate.",
    )
    outcome_metrics: ModelOutcomeDelta = Field(
        ...,
        description="Outcome delta between baseline and candidate.",
    )
    roi_positive: bool = Field(
        ...,
        description=(
            "Whether the pattern demonstrates positive ROI "
            "(cost savings AND quality maintained or improved)."
        ),
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when this record was created.",
    )


__all__: list[str] = ["ModelAttributionRecord"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result produced by ScoringReducer for a single evaluation run.

Ticket: OMN-2927
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.objective.model_score_vector import ModelScoreVector
from omnibase_infra.nodes.node_reward_binder_effect.models.model_evidence_bundle import (
    ModelEvidenceBundle,
)


class ModelEvaluationResult(BaseModel):
    """Result produced by ScoringReducer for a single evaluation run.

    Uses canonical omnibase_core.ModelScoreVector (correctness, safety, cost,
    latency, maintainability, human_time) rather than the stub's per-target
    composite_score/dimensions fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID = Field(..., description="Unique evaluation run ID.")
    objective_id: UUID = Field(..., description="Objective that drove this run.")
    score_vector: ModelScoreVector = Field(
        ...,
        description="Canonical multi-dimensional score vector for this run.",
    )
    evidence_bundle: ModelEvidenceBundle = Field(
        ..., description="Evidence supporting this evaluation."
    )
    policy_state_before: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot before this evaluation.",
    )
    policy_state_after: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot after this evaluation.",
    )


__all__: list[str] = ["ModelEvaluationResult"]

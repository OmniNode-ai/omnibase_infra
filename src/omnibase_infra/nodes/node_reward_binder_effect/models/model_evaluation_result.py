# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result produced by ScoringReducer for a single evaluation run.

Stub model pending OMN-2537 merge (canonical models in omnibase_core).

Ticket: OMN-2552
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_reward_binder_effect.models.model_evidence_bundle import (
    ModelEvidenceBundle,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_score_vector import (
    ModelScoreVector,
)


class ModelEvaluationResult(BaseModel):
    """Result produced by ScoringReducer for a single evaluation run.

    Stub pending OMN-2537 merge into omnibase_core.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID = Field(..., description="Unique evaluation run ID.")
    objective_id: UUID = Field(..., description="Objective that drove this run.")
    score_vectors: tuple[ModelScoreVector, ...] = Field(
        default_factory=tuple,
        description="Per-target score vectors.",
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

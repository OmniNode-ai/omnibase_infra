# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for the RewardBinder EFFECT node.

Ticket: OMN-2552
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_reward_binder_effect.models.model_reward_domain import (
    EvaluationResult,
    ObjectiveSpec,
)


class ModelRewardBinderInput(BaseModel):
    """Input envelope for RewardBinderEffect operations.

    Carries the ``EvaluationResult`` produced by ``ScoringReducer`` together
    with the ``ObjectiveSpec`` used for the run (required for
    ``objective_fingerprint`` computation).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    evaluation_result: EvaluationResult = Field(
        ...,
        description="Evaluation result produced by ScoringReducer.",
    )
    objective_spec: ObjectiveSpec = Field(
        ...,
        description="ObjectiveSpec used for this evaluation run (for fingerprint computation).",
    )


__all__: list[str] = ["ModelRewardBinderInput"]

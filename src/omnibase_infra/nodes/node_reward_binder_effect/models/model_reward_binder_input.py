# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for the RewardBinder EFFECT node.

Ticket: OMN-2552
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_reward_binder_effect.models.model_evaluation_result import (
    ModelEvaluationResult,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_objective_spec import (
    ModelObjectiveSpec,
)


class ModelRewardBinderInput(BaseModel):
    """Input envelope for RewardBinderEffect operations.

    Carries the ``ModelEvaluationResult`` produced by ``ScoringReducer`` together
    with the ``ModelObjectiveSpec`` used for the run (required for
    ``objective_fingerprint`` computation).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    evaluation_result: ModelEvaluationResult = Field(
        ...,
        description="Evaluation result produced by ScoringReducer.",
    )
    objective_spec: ModelObjectiveSpec = Field(
        ...,
        description="ObjectiveSpec used for this evaluation run (for fingerprint computation).",
    )


__all__: list[str] = ["ModelRewardBinderInput"]

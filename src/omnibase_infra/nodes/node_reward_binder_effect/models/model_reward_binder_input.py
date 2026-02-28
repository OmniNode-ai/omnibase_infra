# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for the RewardBinder EFFECT node.

Updated in OMN-2928 to include policy_id and policy_type, enabling the
handler to emit the canonical ModelRewardAssignedEvent shape.

Ticket: OMN-2552, OMN-2928
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_policy_type import EnumPolicyType
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
    ``objective_fingerprint`` computation), plus the policy entity being scored.

    ``policy_id`` and ``policy_type`` are required so the handler can emit
    the canonical ``ModelRewardAssignedEvent`` shape consumable by
    ``omniintelligence/NodePolicyStateReducer``.
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
    policy_id: UUID = Field(
        ...,
        description=(
            "The policy entity ID receiving the reward "
            "(maps to tool_id, pattern_id, model_id, or agent_id)."
        ),
    )
    policy_type: EnumPolicyType = Field(
        ...,
        description="Which policy type this reward applies to.",
    )


__all__: list[str] = ["ModelRewardBinderInput"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A diff to apply, the policy that bounds it and the round applying it."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.declared_state.model_declared_state_apply_policy import (
    ModelDeclaredStateApplyPolicy,
)
from omnibase_infra.models.declared_state.model_declared_state_diff import (
    ModelDeclaredStateDiff,
)


class ModelDeclaredStateApplyRequest(BaseModel):
    """One round's plan for one surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    diff: ModelDeclaredStateDiff
    policy: ModelDeclaredStateApplyPolicy
    round_key: str = Field(min_length=1)


__all__ = ["ModelDeclaredStateApplyRequest"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One reconcile round for one surface."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.models.declared_state.enum_declared_state_reconcile_mode import (
    EnumDeclaredStateReconcileMode,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_policy import (
    ModelDeclaredStateApplyPolicy,
)
from omnibase_infra.models.declared_state.model_declared_state_item import (
    ModelDeclaredStateItem,
)


class ModelDeclaredStateReconcileRequest(BaseModel):
    """The declaration of one surface and how far this round may go."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    surface_kind: EnumDeclaredStateSurfaceKind
    surface_key: str = Field(min_length=1)
    declared: tuple[ModelDeclaredStateItem, ...]
    mode: EnumDeclaredStateReconcileMode
    policy: ModelDeclaredStateApplyPolicy
    round_key: str = Field(min_length=1)


__all__ = ["ModelDeclaredStateReconcileRequest"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one reconcile round found and did."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.declared_state.enum_declared_state_reconcile_mode import (
    EnumDeclaredStateReconcileMode,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_result import (
    ModelDeclaredStateApplyResult,
)


class ModelDeclaredStateReconcileResult(BaseModel):
    """The round's planned operation count and, when it tried, the apply outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_key: str = Field(min_length=1)
    mode: EnumDeclaredStateReconcileMode
    operation_count: int = Field(ge=0)
    unobservable_reason: str | None
    apply: ModelDeclaredStateApplyResult | None = Field(
        description="Null when the round only observed or had nothing to apply."
    )


__all__ = ["ModelDeclaredStateReconcileResult"]

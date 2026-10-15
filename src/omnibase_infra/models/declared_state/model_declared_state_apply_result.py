# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How one apply ended."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.declared_state.enum_declared_state_apply_status import (
    EnumDeclaredStateApplyStatus,
)


class ModelDeclaredStateApplyResult(BaseModel):
    """Operations applied and, for every refusal, the reason in words."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EnumDeclaredStateApplyStatus
    applied_count: int = Field(ge=0)
    refusal: str | None = Field(description="Why the apply stopped; null when applied.")

    @model_validator(mode="after")
    def _refusal_matches_status(self) -> Self:
        """A refusal is named, and an applied result names none."""
        applied = self.status is EnumDeclaredStateApplyStatus.APPLIED
        if applied != (self.refusal is None):
            raise ValueError(
                "refusal is required exactly when the apply did not complete"
            )
        return self


__all__ = ["ModelDeclaredStateApplyResult"]

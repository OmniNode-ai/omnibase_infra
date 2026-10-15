# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The plan one round derives from what it observed."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.models.declared_state.model_declared_state_operation import (
    ModelDeclaredStateOperation,
)


class ModelDeclaredStateDiff(BaseModel):
    """Operations for one surface, planned from one observed generation.

    An unobservable surface carries a reason and no operations: nothing is
    ever applied to a surface the round could not read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    surface_kind: EnumDeclaredStateSurfaceKind
    surface_key: str = Field(min_length=1)
    generation: str = Field(
        min_length=1, description="Surface generation the observation was read at."
    )
    operations: tuple[ModelDeclaredStateOperation, ...]
    unobservable_reason: str | None = Field(
        description="Why the surface could not be diffed; null when it could."
    )

    @model_validator(mode="after")
    def _unobservable_plans_nothing(self) -> Self:
        """An unobservable diff holds no operations."""
        if self.unobservable_reason is not None and self.operations:
            raise ValueError("an unobservable diff cannot carry operations")
        return self


__all__ = ["ModelDeclaredStateDiff"]

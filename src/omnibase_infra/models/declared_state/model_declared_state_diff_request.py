# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one diff compares: the declaration against one round's observations."""

from collections import Counter
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_core.models.config_overlay.model_declared_state_observation import (
    ModelDeclaredStateObservation,
)
from omnibase_infra.models.declared_state.model_declared_state_item import (
    ModelDeclaredStateItem,
)


class ModelDeclaredStateDiffRequest(BaseModel):
    """The declared items and observed rows of exactly one surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    surface_kind: EnumDeclaredStateSurfaceKind
    surface_key: str = Field(min_length=1)
    generation: str = Field(
        min_length=1, description="Surface generation the observations were read at."
    )
    declared: tuple[ModelDeclaredStateItem, ...]
    observations: tuple[ModelDeclaredStateObservation, ...]

    @model_validator(mode="after")
    def _observations_belong_to_the_surface(self) -> Self:
        """Refuse a row from another surface, and two rows for one item."""
        for row in self.observations:
            if (row.surface_kind, row.surface_id) != (
                self.surface_kind,
                self.surface_key,
            ):
                raise ValueError(
                    f"observation of {row.item_key!r} is for surface "
                    f"{row.surface_kind.value}/{row.surface_id}, not "
                    f"{self.surface_kind.value}/{self.surface_key}"
                )
        counts = Counter(row.item_key for row in self.observations)
        repeated = sorted(key for key, count in counts.items() if count > 1)
        if repeated:
            raise ValueError(f"duplicate observation rows for {repeated}")
        return self


__all__ = ["ModelDeclaredStateDiffRequest"]

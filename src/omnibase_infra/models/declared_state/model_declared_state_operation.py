# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One change planned against one item of a surface."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.models.declared_state.enum_declared_state_operation_kind import (
    EnumDeclaredStateOperationKind,
)


class ModelDeclaredStateOperation(BaseModel):
    """Create, update or delete one item; a delete carries no digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EnumDeclaredStateOperationKind
    surface_kind: EnumDeclaredStateSurfaceKind
    surface_key: str = Field(min_length=1)
    item_key: str = Field(min_length=1)
    digest: str | None = Field(
        description="Digest the item must hold after the operation; null for a delete."
    )


__all__ = ["ModelDeclaredStateOperation"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which surface kinds an overlay lets the reconcile node change."""

from pydantic import BaseModel, ConfigDict

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)


class ModelDeclaredStateApplyPolicy(BaseModel):
    """Kinds apply may change; every other kind is observe-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_kinds: frozenset[EnumDeclaredStateSurfaceKind]


__all__ = ["ModelDeclaredStateApplyPolicy"]

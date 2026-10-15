# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Port to the observe effect that reads one surface."""

from typing import Protocol

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.models.declared_state.model_declared_state_item import (
    ModelDeclaredStateItem,
)
from omnibase_infra.models.declared_state.model_declared_state_snapshot import (
    ModelDeclaredStateSnapshot,
)


class ProtocolDeclaredStateObserver(Protocol):
    """Read a surface and report its generation with one row per item."""

    async def observe(
        self,
        surface_kind: EnumDeclaredStateSurfaceKind,
        surface_key: str,
        round_key: str,
        declared: tuple[ModelDeclaredStateItem, ...],
    ) -> ModelDeclaredStateSnapshot:
        """Return a row for every declared item and every undeclared item present."""
        ...


__all__ = ["ProtocolDeclaredStateObserver"]

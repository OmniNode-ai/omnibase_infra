# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Port to one kind of surface, bound per lane by overlay."""

from typing import Protocol

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.models.declared_state.model_declared_state_operation import (
    ModelDeclaredStateOperation,
)


class ProtocolDeclaredStateSurface(Protocol):
    """Read a surface's generation and change one of its items."""

    async def read_generation(
        self,
        surface_kind: EnumDeclaredStateSurfaceKind,
        surface_key: str,
    ) -> str:
        """Return the token that changes whenever the surface changes."""
        ...

    async def apply_operation(
        self,
        operation: ModelDeclaredStateOperation,
        holder: str,
    ) -> str:
        """Make the one change and return the generation that change produced.

        The adapter reports the generation atomically with the change, so an
        edit made by someone else after it is never mistaken for this apply.
        """
        ...


__all__ = ["ProtocolDeclaredStateSurface"]

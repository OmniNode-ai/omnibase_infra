# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-process lease: one holder per (surface kind, surface id)."""

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.declared_state.model_lease_table import ModelLeaseTable


class HandlerDeclaredStateLeaseLocal:
    """Serialize rounds inside one process; a shared store binds the cross-process lease."""

    def __init__(self) -> None:
        self._leases = ModelLeaseTable()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the infrastructure role."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the side-effecting category: holding a lease mutates shared state."""
        return EnumHandlerTypeCategory.EFFECT

    async def acquire(
        self,
        surface_kind: EnumDeclaredStateSurfaceKind,
        surface_key: str,
        holder: str,
    ) -> None:
        """Wait for the surface's lease, then hold it for ``holder``."""
        await self._leases.acquire((surface_kind, surface_key), holder)

    async def release(
        self,
        surface_kind: EnumDeclaredStateSurfaceKind,
        surface_key: str,
        holder: str,
    ) -> None:
        """Release the lease only when ``holder`` is the one holding it."""
        self._leases.release((surface_kind, surface_key), holder)


__all__ = ["HandlerDeclaredStateLeaseLocal"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-process lease: one holder per (surface kind, surface id)."""

import asyncio

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory


class HandlerDeclaredStateLeaseLocal:
    """Serialize rounds inside one process; a shared store binds the cross-process lease."""

    def __init__(self) -> None:
        self._locks: dict[tuple[EnumDeclaredStateSurfaceKind, str], asyncio.Lock] = {}
        self._holders: dict[tuple[EnumDeclaredStateSurfaceKind, str], str] = {}

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
        key = (surface_kind, surface_key)
        await self._locks.setdefault(key, asyncio.Lock()).acquire()
        self._holders[key] = holder

    async def release(
        self,
        surface_kind: EnumDeclaredStateSurfaceKind,
        surface_key: str,
        holder: str,
    ) -> None:
        """Release the lease only when ``holder`` is the one holding it."""
        key = (surface_kind, surface_key)
        if self._holders.get(key) != holder:
            return
        del self._holders[key]
        self._locks[key].release()


__all__ = ["HandlerDeclaredStateLeaseLocal"]

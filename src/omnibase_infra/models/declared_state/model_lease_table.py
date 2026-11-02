# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-process lease table: one holder per (surface kind, surface key) (OMN-21536)."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)

type LeaseKey = tuple[EnumDeclaredStateSurfaceKind, str]


class ModelLeaseTable(BaseModel):
    """The locks and current holders a lease handler serializes rounds with.

    The handler delegates to the table, so it keeps no lock or holder itself.
    """

    model_config = ConfigDict(
        extra="forbid", from_attributes=True, arbitrary_types_allowed=True
    )

    locks: dict[LeaseKey, asyncio.Lock] = Field(
        default_factory=dict, description="One lock per surface."
    )
    holders: dict[LeaseKey, str] = Field(
        default_factory=dict, description="Who holds each locked surface."
    )

    async def acquire(self, key: LeaseKey, holder: str) -> None:
        """Wait for the surface's lock, then record ``holder`` against it."""
        await self.locks.setdefault(key, asyncio.Lock()).acquire()
        self.holders[key] = holder

    def release(self, key: LeaseKey, holder: str) -> None:
        """Release the surface's lock only when ``holder`` is the one holding it."""
        if self.holders.get(key) != holder:
            return
        del self.holders[key]
        self.locks[key].release()


__all__: list[str] = ["LeaseKey", "ModelLeaseTable"]

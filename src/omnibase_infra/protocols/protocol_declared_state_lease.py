# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Port to the per-surface lease that serializes applies."""

from typing import Protocol

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)


class ProtocolDeclaredStateLease(Protocol):
    """One holder per surface; acquire waits until the surface is free."""

    async def acquire(
        self,
        surface_kind: EnumDeclaredStateSurfaceKind,
        surface_key: str,
        holder: str,
    ) -> None:
        """Wait for the surface's lease, then hold it for ``holder``."""
        ...

    async def release(
        self,
        surface_kind: EnumDeclaredStateSurfaceKind,
        surface_key: str,
        holder: str,
    ) -> None:
        """Release the lease ``holder`` holds."""
        ...


__all__ = ["ProtocolDeclaredStateLease"]

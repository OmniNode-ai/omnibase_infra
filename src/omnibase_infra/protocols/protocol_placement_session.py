# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Port to a runtime's placement membership session (OMN-22132)."""

from typing import Protocol


class ProtocolPlacementSession(Protocol):
    """Lease, membership key and local self-fence deadline of one runtime."""

    @property
    def runtime_id(self) -> str:
        """The runtime id this session registers membership under."""
        ...

    @property
    def node_names(self) -> tuple[str, ...]:
        """The node names this session registers membership for."""
        ...

    @property
    def lease_id(self) -> int:
        """The open lease, ``0`` when no session is open."""
        ...

    async def open(self) -> None:
        """Grant the lease and register membership; raises when the store is unreachable."""
        ...

    async def keepalive(self) -> bool:
        """Renew once; False when the lease is gone."""
        ...

    def drop(self) -> None:
        """Forget the session locally."""
        ...

    async def close(self) -> None:
        """Revoke the lease so membership and leader keys go at once."""
        ...


__all__ = ["ProtocolPlacementSession"]

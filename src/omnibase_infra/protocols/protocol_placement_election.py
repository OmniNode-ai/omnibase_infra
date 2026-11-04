# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Port to one runtime's candidacy for one node's primary role (OMN-22132)."""

from typing import Protocol


class ProtocolPlacementElection(Protocol):
    """Campaign, verify and step down for one ``primary`` node."""

    @property
    def epoch(self) -> int:
        """The epoch this runtime won with, ``0`` when it is not primary."""
        ...

    def acting(self) -> bool:
        """Primary and before the local self-fence deadline."""
        ...

    async def campaign_once(self) -> bool:
        """Try to take the free leader key; True when this runtime is primary."""
        ...

    async def verify_holder(self) -> bool:
        """Step down when the leader key is not this runtime's at this epoch."""
        ...

    def step_down(self) -> None:
        """Stop acting as primary."""
        ...


__all__ = ["ProtocolPlacementElection"]

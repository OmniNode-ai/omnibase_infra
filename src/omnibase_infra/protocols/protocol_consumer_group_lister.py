# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for a bus that lists the consumer groups it can see.

OMN-19998. Auto-wiring joins each contract topic under a consumer group id
built from the contract version, so a version bump starts the new group at
the bus default and drops every record published in between. Seeding the
successor group requires the ids of the predecessor groups, which only a bus
that keeps group state knows; the version handover asks it directly. Buses
that keep no group state (the in-memory bus) have no predecessor and the
handover does nothing for them.

The method name is deliberately distinctive: ``@runtime_checkable`` checks
member presence, not signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["ProtocolConsumerGroupLister"]


@runtime_checkable
class ProtocolConsumerGroupLister(Protocol):
    """A bus that reports every consumer group id it can see."""

    async def list_consumer_groups(self) -> Sequence[str]:
        """Every group id the bus can see, in any order.

        The answer must include the retired groups, not only the live ones:
        the caller matches them against the successor's group id, so a
        listing that omits stopped groups would skip exactly the predecessor
        whose position the handover is there to save.
        """
        ...

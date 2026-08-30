# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one operator read of the cloud ledger projection (OMN-17205).

Five members, not two. ``beta/GOAL.md`` row 0b's probe is only useful if it can
distinguish the states that look alike from outside the cluster:

  * a healthy pipeline with no row for this correlation id yet,
  * a projection that does not exist on that plane at all (the leg-5 sink gap),
  * a credential the control plane refused,
  * a control plane that is there but cannot answer.

Collapsing any of those into "empty" produces a probe that reports the same
thing whether the chain is healthy-but-idle or dead -- which is precisely the
class of goal row that cannot catch a drop (RC-L).

Ticket: OMN-17205
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumCloudLedgerVerdict(str, Enum):
    """What the cloud ledger said about one correlation id."""

    FOUND = "found"
    """At least one projected row carries this correlation id. The only pass."""

    NOT_FOUND = "not_found"
    """The projection exists and holds no row for this correlation id."""

    PROJECTION_ABSENT = "projection_absent"
    """The projection table does not exist on this plane yet."""

    UNAUTHENTICATED = "unauthenticated"
    """The credential was absent, refused, or could not be minted."""

    UNAVAILABLE = "unavailable"
    """The control plane was reached but could not answer, or was not reached."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumCloudLedgerVerdict"]

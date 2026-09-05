# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which enumeration arm put a candidate in front of the sweep (OMN-17342)."""

from __future__ import annotations

from enum import StrEnum


class EnumEvidenceAutocloseArm(StrEnum):
    """The selection path a considered (companion, ticket) pair arrived on.

    Recorded on every outcome because the two arms answer different questions
    and their results are not interchangeable in an audit. FORWARD is the
    freshness arm: it sees a companion once, in the window immediately after it
    merges, and its coverage claim is "nothing that merged recently was
    missed". BACKFILL is the drain arm: it re-offers older companions on a
    rotating slice, and its coverage claim is only "this slice was examined on
    this tick" — a ticket absent from a backfill run is not a ticket that was
    refused, it is one whose turn has not come round yet.

    Reporting them apart is what makes OMN-17342's AC5 measurable at all: the
    standing-backlog population can only be shown to be draining if a receipt
    says which decisions came from the arm that reaches it.
    """

    FORWARD = "forward"
    BACKFILL = "backfill"


__all__ = ["EnumEvidenceAutocloseArm"]

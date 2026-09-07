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
    # OMN-16106 Item 1. The RESTRICTIVE arm. Neither a window nor a slice: the
    # caller named the ticket and its newest merged companion was resolved
    # directly, so this arm makes NO coverage claim at all — not "nothing
    # recent was missed" and not "this slice was examined". It claims only
    # "these named tickets were adjudicated on this run".
    #
    # It is a third value rather than a flag on the outcome because the two
    # existing arms are read as coverage evidence. An offered decision recorded
    # as FORWARD would make a receipt assert a freshness window the run never
    # enumerated, which is the same class of claim as a label that reads as a
    # fact and is not one.
    OFFER = "offer"


__all__ = ["EnumEvidenceAutocloseArm"]

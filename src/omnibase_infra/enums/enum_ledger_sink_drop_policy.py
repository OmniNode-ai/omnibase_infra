# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Enum for ledger sink drop policy."""

from __future__ import annotations

from enum import Enum


class EnumLedgerSinkDropPolicy(str, Enum):
    """Drop policy when sink queue is full or latency exceeded.

    Attributes:
        DROP_OLDEST: Drop oldest events in queue to make room (lossy but bounded).
        DROP_NEWEST: Drop incoming event (current call's event is lost).
        BLOCK: Block until space available (violates latency budget - use with caution).
        RAISE: Raise exception (caller must handle).
    """

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    RAISE = "raise"


__all__ = ["EnumLedgerSinkDropPolicy"]

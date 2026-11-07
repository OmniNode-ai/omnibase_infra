# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle phases of a bus subscription (OMN-21229)."""

from __future__ import annotations

from enum import Enum


class EnumSubscriptionPhase(str, Enum):
    """Where a subscription handle is in pause, drain, stop.

    The phases only move forward, in this order: ``ACTIVE`` delivers records,
    ``PAUSED`` takes no new record, ``DRAINED`` has finished or released every
    record it had already fetched, ``STOPPED`` has removed its callback.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    DRAINED = "drained"
    STOPPED = "stopped"


__all__ = ["EnumSubscriptionPhase"]

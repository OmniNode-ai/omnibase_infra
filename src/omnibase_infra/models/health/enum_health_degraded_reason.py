# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed reasons a ``/health`` answer reports ``degraded`` (OMN-26316)."""

from __future__ import annotations

from enum import Enum


class EnumHealthDegradedReason(str, Enum):
    """Why ``/health`` answered ``degraded`` instead of the runtime's own verdict.

    EVENT_LOOP_UNRESPONSIVE: the runtime's asyncio event loop ran no task for
        longer than the acceptor's wait budget, so the probe was answered from
        the acceptor thread instead of the loop-hosted health handler.
    """

    EVENT_LOOP_UNRESPONSIVE = "event_loop_unresponsive"


__all__ = ["EnumHealthDegradedReason"]

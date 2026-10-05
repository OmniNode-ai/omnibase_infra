# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Observable phases of one delegation's execution (OMN-19452).

Separate phase names let callers distinguish startup and transport latency
from the time spent waiting for a terminal event.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["EnumDelegatePhase"]


class EnumDelegatePhase(StrEnum):
    """Phases whose completed spans contribute to a delegation's durations."""

    STARTUP = "startup"
    LOCUS_PROBE = "locus_probe"
    BUS_CONNECT = "bus_connect"
    REPLY_SUBSCRIBE = "reply_subscribe"
    PUBLISH = "publish"
    TERMINAL_WAIT = "terminal_wait"

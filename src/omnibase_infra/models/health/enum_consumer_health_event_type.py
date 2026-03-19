# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer health event type classification enum.

Ticket: OMN-5511
"""

from __future__ import annotations

from enum import StrEnum


class EnumConsumerHealthEventType(StrEnum):
    """Classification of consumer health events."""

    HEARTBEAT_FAILURE = "heartbeat_failure"
    SESSION_TIMEOUT = "session_timeout"
    REBALANCE_START = "rebalance_start"
    REBALANCE_COMPLETE = "rebalance_complete"
    CONSUMER_STOPPED = "consumer_stopped"
    CONSUMER_STARTED = "consumer_started"
    POLL_TIMEOUT = "poll_timeout"
    CONNECTION_LOST = "connection_lost"


__all__ = ["EnumConsumerHealthEventType"]

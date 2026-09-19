# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a consumer group was classified as stalled, or why it was not (OMN-18640).

The reasons are deliberately distinct rather than a single boolean: a group with
no assignment and a group holding an assignment it is not draining have the same
visible symptom (no records, growing backlog) and the same remedy here, but they
arise from different broker-side states and a reader of the emitted event needs
to tell them apart.
"""

from __future__ import annotations

from enum import StrEnum


class EnumConsumerStallReason(StrEnum):
    """Classification of a single consumer stall evaluation."""

    NOT_STALLED_PROGRESSING = "not_stalled_progressing"
    """Records were delivered inside the evaluation window."""

    NOT_STALLED_IDLE = "not_stalled_idle"
    """No records, but the fetch position is at the end of every assigned
    partition: there is nothing to consume, which is health, not a wedge."""

    NOT_STALLED_BROKER_UNREACHABLE = "not_stalled_broker_unreachable"
    """The end-offset probe failed, so the broker itself is unreachable. A
    client-side rejoin cannot fix a broker that is down, and recreating into an
    unreachable broker would churn. Fails toward leaving the consumer alone."""

    NOT_STALLED_WITHIN_COOLDOWN = "not_stalled_within_cooldown"
    """The stall signature holds but this group rejoined too recently. Bounds
    the recreate rate so a persistent fault cannot become a recreate loop."""

    STALLED_BACKLOG_NOT_DRAINING = "stalled_backlog_not_draining"
    """Partitions are assigned, the broker answers an end-offset probe, the end
    offset is ahead of the fetch position, and no record has been delivered for
    the configured window. This is the 2026-09-17 and 2026-09-18 signature."""

    STALLED_NO_ASSIGNMENT = "stalled_no_assignment"
    """The consumer holds no partition assignment while the broker is reachable
    and the configured window has elapsed. This is the ``Empty`` group shape
    recorded on OMN-18640 as the second outage class."""


__all__ = ["EnumConsumerStallReason"]

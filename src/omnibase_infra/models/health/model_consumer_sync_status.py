# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether one consumer group this runtime owns is in sync (OMN-18640 AC1).

**Why a second model beside ``ModelConsumerStallVerdict``.** That verdict
answers one question for the recovery path: *may I recreate this consumer
now?* Its vocabulary is shaped entirely by that question, and two of its
answers are wrong for readiness. ``NOT_STALLED_WITHIN_COOLDOWN`` means "still
wedged, but I rejoined too recently to try again" -- the cooldown bounds the
REMEDY, it does not make the group healthy, yet a readiness surface that
inherited that word would read green for the whole of every cooldown. And a
verdict is per-evaluation, so it cannot say how long the current stall has
been running, which is the fact readiness is actually judged on.

**Why the window is longer than the stall window.** The recovery path acts at
``consumer_stall_seconds`` (120s) after ``required_consecutive_stalls`` (3)
confirmations, and retries no faster than ``consumer_rejoin_cooldown_seconds``
(300s). A readiness surface that flipped on the same timescale would declare
the runtime unhealthy while the cheapest remedy was still in the middle of
working, and the deploy agent's force-recreate (OMN-18640 AC7) would then
recreate a container whose consumer was about to fix itself. Readiness reports
the wedge the self-heal did NOT fix.

**What ``ready`` is not.** It is not a field. It is derived from the numbers
beside it, for the same reason ``ModelConsumerStallVerdict.should_rejoin`` and
``ModelLabPassReceipt``'s verdict are derived: this ticket exists because three
separate surfaces stated health that they had not measured, for fifty minutes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)


class ModelConsumerSyncStatus(BaseModel):
    """Sync state of one ``(topic, consumer group)`` this runtime consumes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., description="Topic this consumer is subscribed to.")
    consumer_group: str = Field(..., description="Effective Kafka consumer group id.")

    evaluated: bool = Field(
        ...,
        description=(
            "True once at least one stall evaluation has run for this group. "
            "False means no measurement exists yet -- distinct from a measured "
            "green, so a reader can tell 'proven in sync' from 'nothing known "
            "yet'. A never-evaluated group is READY: at boot there is nothing "
            "to be red about."
        ),
    )
    reason: EnumConsumerStallReason = Field(
        ...,
        description="Classification from the most recent stall evaluation.",
    )
    backlog_records: int = Field(
        ...,
        description=(
            "Records between this consumer's fetch position and the partition "
            "LEADERS' log end offsets, summed over its assignment. The lag "
            "half of the dimension's evidence."
        ),
        ge=0,
    )
    seconds_since_last_record: float = Field(
        ...,
        description=(
            "Monotonic seconds since this consumer last delivered a record. "
            "The last-advance half of the dimension's evidence."
        ),
        ge=0.0,
    )
    stalled_seconds: float = Field(
        ...,
        description=(
            "How long the CURRENT stall has run: seconds since the first "
            "evaluation in the present unbroken run of stall signatures. Zero "
            "whenever the group is not currently carrying one. Counted across "
            "rejoins and across the cooldown, because a group that wedges, "
            "rejoins into the same wedge and wedges again has not recovered."
        ),
        ge=0.0,
    )
    assigned_partitions: int = Field(
        ..., description="Partitions currently assigned to this consumer.", ge=0
    )
    broker_reachable: bool = Field(
        ...,
        description=(
            "Whether the last end-offset probe against the partition leaders "
            "succeeded. A group whose brokers are unreachable is NOT reported "
            "out of sync: that is a broker fault, and naming the consumer "
            "would point recovery at the wrong container."
        ),
    )
    rejoin_count: int = Field(
        ...,
        description="Forced rejoins this supervisor has attempted for this group.",
        ge=0,
    )
    last_rejoin_failed: bool = Field(
        ...,
        description=(
            "True when the most recent forced rejoin could not start a "
            "replacement consumer, and no record has been delivered since. "
            "Recorded separately from the stall because the remedy failing and "
            "the group being behind are different facts: a rejoin resets the "
            "stall evidence, so a surface that watched only the stall would go "
            "green on a recovery that did not happen. Cleared by flow "
            "resuming, which is the only thing that proves resolution."
        ),
    )
    unready_after_seconds: float = Field(
        ...,
        description=(
            "The declared window this status was judged against, carried so "
            "the evidence is self-describing -- a reader must not need the "
            "runtime's configuration to interpret the verdict."
        ),
        gt=0.0,
    )

    @property
    def stalled(self) -> bool:
        """True when the group currently carries a stall signature."""
        return self.stalled_seconds > 0.0

    @property
    def ready(self) -> bool:
        """Derived. False when a stall has outlived the declared window, or
        when the recovery path tried to fix it and could not."""
        if self.last_rejoin_failed:
            return False
        return self.stalled_seconds < self.unready_after_seconds


__all__ = ["ModelConsumerSyncStatus"]

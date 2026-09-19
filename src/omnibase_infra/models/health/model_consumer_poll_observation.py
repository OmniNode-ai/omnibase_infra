# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One evaluation's worth of facts about a consumer's fetch progress (OMN-18640).

Every field here is MEASURED at the seam that owns it and handed to a pure
decision function. Nothing in this model is asserted by a caller that wants a
particular verdict: ``broker_reachable`` is the outcome of an end-offset probe
that was actually issued, and ``backlog_records`` is the difference between the
partition leaders' answer and the consumer's own fetch position.

The separation matters because the stall this model describes is invisible to
every other surface. On 2026-09-17 and again on 2026-09-18 the wedged consumer
reported ``Stable`` with one member and an assigned partition, the container
reported ``Up``, and the broker reported healthy -- while no record moved for
30 and then 97 minutes. The only two facts that disagreed were the fetch
position, which did not advance, and the log-end offset, which did. Those two
are what this model carries.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConsumerPollObservation(BaseModel):
    """Measured state of one consumer at one evaluation point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., description="Topic this consumer is subscribed to.")
    consumer_group: str = Field(..., description="Effective Kafka consumer group id.")

    seconds_since_last_record: float = Field(
        ...,
        description=(
            "Monotonic seconds since this consumer last delivered a record. "
            "Reset to zero by any successful fetch, so an active consumer never "
            "accumulates a window regardless of how slow its handlers are."
        ),
        ge=0.0,
    )
    assigned_partitions: int = Field(
        ...,
        description="Partitions currently assigned to this consumer.",
        ge=0,
    )
    broker_reachable: bool = Field(
        ...,
        description=(
            "True when the end-offset probe against the partition LEADERS "
            "succeeded. Deliberately independent of the group coordinator: the "
            "wedge this model exists for is a client that can reach every "
            "leader and cannot reach its coordinator."
        ),
    )
    backlog_records: int = Field(
        ...,
        description=(
            "Sum over assigned partitions of (log end offset - fetch position). "
            "Zero means the consumer is caught up and silence is idleness. "
            "Positive and not shrinking across evaluations means silence is a "
            "wedge. Zero when the broker was unreachable, which is why the "
            "verdict must consult broker_reachable first."
        ),
        ge=0,
    )
    seconds_since_last_rejoin: float | None = Field(
        default=None,
        description=(
            "Monotonic seconds since this supervisor last forced a rejoin for "
            "this group, or None when it never has. Bounds the recreate rate."
        ),
        ge=0.0,
    )


__all__ = ["ModelConsumerPollObservation"]

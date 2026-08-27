# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-consumer throughput window deltas carried on the node heartbeat (OMN-16777).

Every liveness signal the platform had before this measured *connectedness* —
group membership, process liveness, container health, consumer LAG.  None of
them measured **throughput across a seam**, which is the only question that
separates a dead consumer from a correctly-idle one.  ``node_gateway_link_health
_projection_compute`` on the ``.201`` dev lane was Stable, LAG 0, current-offset
15,750 with an output topic at LOG-END-OFFSET 0: 15,750 messages in, zero out,
and every check green (OMN-16755).

These models are the **raw counters only**.  They deliberately carry no verdict:
``FLOWING`` / ``STALLED`` / ``STARVED`` / ``IDLE`` is derived in the projection
(``node_projection_consumer_flow`` in omnimarket), never stamped on the producing
event.  A producer that classifies its own health is a producer that can lie
about it, and the envelope-purity gate forbids it.

Time is always INJECTED.  Neither model has a ``default_factory`` clock and
nothing in the accumulator that fills them calls ``now()`` — ``window_start`` and
``window_end`` are producer-assigned event time, handed in by the heartbeat tick.

Related: OMN-16777 (this ticket), epic OMN-16776, OMN-16755 / OMN-16754 /
OMN-16690 / OMN-16767 (the four failures this exists to surface).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelConsumerFlowDelta(BaseModel):
    """One (consumer_group, topic) throughput delta over one heartbeat window.

    A row is emitted for every *registered* subscription every window, including
    windows in which nothing moved.  That is load-bearing: a zero row means
    "this consumer was alive and took nothing", which is a fact.  The ABSENCE of
    a row means "we do not know", which is a different fact and must never be
    materialized as zero (OMN-16777 AC5).

    Attributes:
        consumer_group: The Kafka consumer group id the subscription joined.
        topic: The subscribed topic this delta counts.
        node_id: The node whose process accumulated these counters.
        window_start: Producer-assigned event-time start of the window
            (exclusive of the previous window; equal to the previous
            ``window_end``).
        window_end: Producer-assigned event-time end of the window.
        window_sequence: Monotonically increasing per-process window counter.
            A gap in this sequence for a given ``node_id`` is a DROPPED window
            and must materialize as ``UNKNOWN``, never as zero traffic.
        messages_in: Envelopes handed to the dispatch engine.
        messages_out: Envelopes successfully published by the handler's result.
        messages_dlq: Envelopes routed to a DLQ or the platform quarantine sink.
        handler_errors: Dispatches whose handler raised.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    consumer_group: str = Field(
        ..., min_length=1, description="Consumer group id of the subscription"
    )
    topic: str = Field(..., min_length=1, description="Subscribed topic")
    node_id: UUID = Field(..., description="Node whose process accumulated these")

    window_start: datetime = Field(..., description="Event-time window start")
    window_end: datetime = Field(..., description="Event-time window end")
    window_sequence: int = Field(
        ..., ge=0, description="Monotonic per-process window counter; gap => UNKNOWN"
    )

    messages_in: int = Field(default=0, ge=0)
    messages_out: int = Field(default=0, ge=0)
    messages_dlq: int = Field(default=0, ge=0)
    handler_errors: int = Field(default=0, ge=0)

    @field_validator("window_start", "window_end")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return validate_timezone_aware_datetime(v)

    @model_validator(mode="after")
    def _window_is_ordered(self) -> ModelConsumerFlowDelta:
        if self.window_end < self.window_start:
            raise ValueError(
                f"window_end {self.window_end.isoformat()} precedes window_start "
                f"{self.window_start.isoformat()}"
            )
        return self


__all__ = ["ModelConsumerFlowDelta"]

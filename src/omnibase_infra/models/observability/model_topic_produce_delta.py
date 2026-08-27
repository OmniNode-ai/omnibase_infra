# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-topic production tally carried on the node heartbeat (OMN-16777).

The evidence that separates ``STARVED`` from ``IDLE``.  See
``model_consumer_flow_delta`` for the full rationale.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelTopicProduceDelta(BaseModel):
    """Count of envelopes this process PUBLISHED to a topic in one window.

    This is the evidence that separates ``STARVED`` from ``IDLE``.  A consumer
    with ``messages_in == 0`` is only starved if something upstream was
    *actually producing*; without that evidence, zero-in is idle and calling it
    starved is the alert-storm failure OMN-16777 AC4 exists to forbid.

    The tally is produced from the runtime's own publish seam
    (``DispatchResultApplier``), not from a broker query — reading the broker on
    a timer would be exactly the poller this ticket forbids.

    Consequence, stated rather than hidden: for a topic the platform never
    publishes to (an external MSK ingress leg, for example), there is no
    upstream evidence on this rail at all.  The projection records that as
    ``upstream_evidence = NONE`` and reports ``IDLE``; it does not guess
    ``STARVED``.

    Attributes:
        topic: The destination topic published to.
        node_id: The node whose process published.
        window_start: Producer-assigned event-time start of the window.
        window_end: Producer-assigned event-time end of the window.
        window_sequence: Monotonic per-process window counter.
        messages_produced: Envelopes successfully published to ``topic``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., min_length=1, description="Destination topic")
    node_id: UUID = Field(..., description="Node whose process published")

    window_start: datetime = Field(..., description="Event-time window start")
    window_end: datetime = Field(..., description="Event-time window end")
    window_sequence: int = Field(..., ge=0)

    messages_produced: int = Field(default=0, ge=0)

    @field_validator("window_start", "window_end")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return validate_timezone_aware_datetime(v)


__all__ = ["ModelTopicProduceDelta"]

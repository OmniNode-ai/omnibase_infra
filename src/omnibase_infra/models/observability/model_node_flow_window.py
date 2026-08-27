# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One heartbeat window's complete flow report for one node process (OMN-16777).

See ``model_consumer_flow_delta`` for the full rationale.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.observability.model_consumer_flow_delta import (
    ModelConsumerFlowDelta,
)
from omnibase_infra.models.observability.model_topic_produce_delta import (
    ModelTopicProduceDelta,
)
from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelNodeFlowWindow(BaseModel):
    """One heartbeat window's complete flow report for one node process.

    Carried on ``ModelNodeHeartbeatEvent.flow_window``.  Riding the heartbeat is
    load-bearing rather than stylistic: the flow signal must die with the thing
    it measures.  A separate poller keeps reporting on a dead runtime; a
    heartbeat that stops arriving is itself the signal (OMN-16776 §3.2).

    Attributes:
        node_id: The reporting node.
        window_start: Event-time start of the window this report covers.
        window_end: Event-time end of the window.
        window_sequence: Monotonic per-process counter for THIS node_id.
        consumer_deltas: One entry per registered (consumer_group, topic).
        produce_deltas: One entry per topic published to in the window.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: UUID
    window_start: datetime
    window_end: datetime
    window_sequence: int = Field(..., ge=0)
    consumer_deltas: tuple[ModelConsumerFlowDelta, ...] = ()
    produce_deltas: tuple[ModelTopicProduceDelta, ...] = ()

    @field_validator("window_start", "window_end")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return validate_timezone_aware_datetime(v)


__all__ = ["ModelNodeFlowWindow"]

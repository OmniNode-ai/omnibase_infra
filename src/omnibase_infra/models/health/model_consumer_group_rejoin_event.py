# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runtime event recording a forced consumer-group rejoin (OMN-18640).

This is the durable, machine-readable record that a readiness dimension reads
to answer "did this runtime wedge, and did it recover itself?". It is a typed
event and deliberately NOT a log line: the 2026-09-17 and 2026-09-18 outages
each produced more than twenty thousand log lines naming the fault and no
surface a gate could read, which is the specific failure this record removes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)


class ModelConsumerGroupRejoinEvent(BaseModel):
    """One forced rejoin of one consumer group, with the evidence behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4, description="Unique event id.")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the rejoin was ordered.",
    )
    topic: str = Field(..., description="Topic whose consumer was recreated.")
    consumer_group: str = Field(..., description="Effective Kafka consumer group id.")
    reason: EnumConsumerStallReason = Field(
        ..., description="The stall class that ordered this rejoin."
    )
    stalled_seconds: float = Field(
        ...,
        description="Seconds without a delivered record when the rejoin fired.",
        ge=0.0,
    )
    backlog_records: int = Field(
        ...,
        description="Records behind the log end offset when the rejoin fired.",
        ge=0,
    )
    consecutive_stalls: int = Field(
        ...,
        description="Confirmations accumulated before the rejoin was ordered.",
        ge=1,
    )
    rejoin_succeeded: bool = Field(
        ...,
        description=(
            "Whether the replacement consumer started. A failed recreate is "
            "recorded too: a record that exists only on success cannot "
            "distinguish a failed recovery from one that never ran."
        ),
    )
    failure_detail: str = Field(
        default="",
        description="Error text when rejoin_succeeded is False; empty otherwise.",
        max_length=500,
    )


__all__ = ["ModelConsumerGroupRejoinEvent"]

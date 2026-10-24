# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed alarm for an unsuccessful consumer recovery episode (OMN-21337)."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelConsumerRejoinAlarm(BaseModel):
    """One alarm per episode, retained independently of health emission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4, description="Unique alarm event id.")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC time the alarm was raised.",
    )
    topic: str = Field(..., description="Topic whose consumer could not recover.")
    consumer_group: str = Field(..., description="Effective Kafka consumer group id.")
    broker: str = Field(..., description="Sanitized bootstrap servers.")
    reason: str = Field(..., description="Recovery failure classification.")
    coordinator_unavailable_seconds: float = Field(
        ..., ge=0, description="Seconds without a coordinator while leaders answered."
    )
    failed_rejoins: int = Field(
        ..., ge=1, description="Unsuccessful rejoin attempts in this episode."
    )
    message: str = Field(..., description="Operator-facing recovery alarm message.")


__all__ = ["ModelConsumerRejoinAlarm"]

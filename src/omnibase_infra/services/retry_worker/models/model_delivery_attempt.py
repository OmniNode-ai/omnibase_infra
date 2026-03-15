# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Delivery attempt model for retry worker.

Represents a row from the delivery_attempts table that tracks notification
delivery state, retry scheduling, and failure history.

Related Tickets:
    - OMN-1454: Implement RetryWorker for subscription notification delivery
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.retry_worker.models.enum_delivery_status import (
    EnumDeliveryStatus,
)


class ModelDeliveryAttempt(BaseModel):
    """A single notification delivery attempt record.

    Maps to a row in the delivery_attempts table. Tracks the state of
    a notification delivery including retry scheduling and failure history.

    Attributes:
        id: Unique identifier for this delivery attempt.
        subscription_id: The subscription that triggered this notification.
        notification_payload: JSON-serialized notification content.
        status: Current delivery status.
        attempt_count: Number of delivery attempts made so far.
        max_attempts: Maximum number of retry attempts allowed.
        next_retry_at: Scheduled time for the next retry attempt.
        last_error: Error message from the most recent failed attempt.
        created_at: When this delivery attempt was first created.
        updated_at: When this delivery attempt was last modified.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    id: UUID = Field(..., description="Unique identifier for this delivery attempt.")
    subscription_id: UUID = Field(
        ..., description="The subscription that triggered this notification."
    )
    notification_payload: str = Field(
        ..., description="JSON-serialized notification content."
    )
    status: EnumDeliveryStatus = Field(..., description="Current delivery status.")
    attempt_count: int = Field(
        default=0,
        ge=0,
        description="Number of delivery attempts made so far.",
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of retry attempts allowed.",
    )
    next_retry_at: datetime | None = Field(
        default=None,
        description="Scheduled time for the next retry attempt.",
    )
    last_error: str = Field(
        default="",
        description="Error message from the most recent failed attempt.",
    )
    created_at: datetime = Field(
        ..., description="When this delivery attempt was first created."
    )
    updated_at: datetime = Field(
        ..., description="When this delivery attempt was last modified."
    )


__all__ = ["ModelDeliveryAttempt"]

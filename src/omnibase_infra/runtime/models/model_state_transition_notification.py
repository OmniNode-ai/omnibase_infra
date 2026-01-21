# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""State Transition Notification Model.

Provides the Pydantic model for state transition notifications stored in the
outbox table and published to event buses.

This model represents a notification that should be published when a projection
state transition occurs. The outbox pattern ensures exactly-once delivery by
storing notifications in the same database transaction as the projection write.

Concurrency Safety:
    This model is immutable (frozen=True) after creation, making it thread-safe
    and coroutine-safe for concurrent read access.

Related Tickets:
    - OMN-1139: TransitionNotificationOutbox implementation

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelStateTransitionNotification(BaseModel):
    """State transition notification for outbox-based delivery.

    Represents a notification that should be published when a state transition
    occurs in a projection. The notification contains the transition details
    needed by downstream consumers to react to state changes.

    Attributes:
        notification_id: Unique identifier for this notification (UUID4).
        aggregate_type: Type of aggregate (e.g., "registration", "node").
        aggregate_id: Unique identifier of the aggregate that transitioned.
        from_state: Previous state before the transition (nullable for creation).
        to_state: New state after the transition.
        event_type: Type of event that triggered the transition.
        event_id: Unique identifier of the triggering event.
        correlation_id: Correlation ID for distributed tracing.
        occurred_at: Timestamp when the transition occurred.
        metadata: Optional additional metadata for the notification.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> notification = ModelStateTransitionNotification(
        ...     notification_id=uuid4(),
        ...     aggregate_type="registration",
        ...     aggregate_id=uuid4(),
        ...     from_state="pending_registration",
        ...     to_state="awaiting_ack",
        ...     event_type="NodeRegistrationAccepted",
        ...     event_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     occurred_at=datetime.now(UTC),
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    notification_id: UUID = Field(
        ...,
        description="Unique identifier for this notification (UUID4)",
    )
    aggregate_type: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Type of aggregate (e.g., 'registration', 'node')",
    )
    aggregate_id: UUID = Field(
        ...,
        description="Unique identifier of the aggregate that transitioned",
    )
    from_state: str | None = Field(
        default=None,
        max_length=128,
        description="Previous state before the transition (nullable for creation)",
    )
    to_state: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="New state after the transition",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Type of event that triggered the transition",
    )
    event_id: UUID = Field(
        ...,
        description="Unique identifier of the triggering event",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    occurred_at: datetime = Field(
        ...,
        description="Timestamp when the transition occurred",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional additional metadata for the notification",
    )

    def to_topic_key(self) -> str:
        """Generate a topic partition key for Kafka publishing.

        Returns:
            String key in format "{aggregate_type}:{aggregate_id}" for
            consistent partitioning of related notifications.

        Example:
            >>> notification.to_topic_key()
            'registration:550e8400-e29b-41d4-a716-446655440000'
        """
        return f"{self.aggregate_type}:{self.aggregate_id}"


__all__: list[str] = [
    "ModelStateTransitionNotification",
]

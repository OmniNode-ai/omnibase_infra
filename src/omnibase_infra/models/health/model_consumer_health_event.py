# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer health event model for Layer 1 health pipeline.

Structured event model emitted by ConsumerHealthEmitter when consumer
lifecycle events occur (heartbeat failures, rebalances, session timeouts).
The fingerprint field enables deduplication and incident grouping in the
triage node.

Related Tickets:
    - OMN-5511: Create consumer health event models and enums
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_consumer_health_event_type import (
    EnumConsumerHealthEventType,
)
from omnibase_infra.models.health.enum_consumer_health_severity import (
    EnumConsumerHealthSeverity,
)


def _compute_fingerprint(
    consumer_identity: str,
    event_type: str,
    topic: str,
) -> str:
    """Compute a stable fingerprint for deduplication and incident grouping.

    Args:
        consumer_identity: Unique consumer identifier.
        event_type: The event type string.
        topic: The Kafka topic the consumer is subscribed to.

    Returns:
        Hex digest fingerprint string.
    """
    raw = f"{consumer_identity}:{event_type}:{topic}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ModelConsumerHealthEvent(BaseModel):
    """Structured consumer health event for Kafka emission.

    Emitted by ConsumerHealthEmitter when consumer lifecycle events
    occur (heartbeat failures, rebalances, session timeouts, etc.).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Identity
    event_id: UUID = Field(
        default_factory=uuid4, description="Unique event identifier."
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing across services.",
    )
    consumer_identity: str = Field(
        ..., description="Unique identifier for the consumer instance."
    )
    consumer_group: str = Field(..., description="Kafka consumer group ID.")
    topic: str = Field(
        ..., description="Primary Kafka topic the consumer is subscribed to."
    )

    # Classification
    event_type: EnumConsumerHealthEventType = Field(
        ..., description="Type of consumer health event."
    )
    severity: EnumConsumerHealthSeverity = Field(
        ..., description="Severity level of the event."
    )

    # Fingerprint for deduplication
    fingerprint: str = Field(
        ...,
        description="Stable hash for deduplication and incident grouping.",
    )

    # Rebalance metrics (populated for rebalance events)
    rebalance_duration_ms: int | None = Field(
        default=None,
        description="Duration of rebalance in milliseconds (for REBALANCE_COMPLETE).",
    )
    partitions_assigned: int | None = Field(
        default=None,
        description="Number of partitions assigned after rebalance.",
    )
    partitions_revoked: int | None = Field(
        default=None,
        description="Number of partitions revoked during rebalance.",
    )

    # Error context
    error_message: str = Field(
        default="",
        description="Error message if applicable. Empty if no error.",
    )
    error_type: str = Field(
        default="",
        description="Exception type name if applicable. Empty if no error.",
    )

    # Metadata
    hostname: str = Field(
        default="", description="Hostname of the machine running the consumer."
    )
    service_label: str = Field(
        default="", description="Display name of the service owning the consumer."
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the event was emitted.",
    )

    @classmethod
    def create(
        cls,
        *,
        consumer_identity: str,
        consumer_group: str,
        topic: str,
        event_type: EnumConsumerHealthEventType,
        severity: EnumConsumerHealthSeverity,
        correlation_id: UUID | None = None,
        rebalance_duration_ms: int | None = None,
        partitions_assigned: int | None = None,
        partitions_revoked: int | None = None,
        error_message: str = "",
        error_type: str = "",
        hostname: str = "",
        service_label: str = "",
    ) -> ModelConsumerHealthEvent:
        """Factory method that auto-computes fingerprint.

        Args:
            consumer_identity: Unique consumer identifier.
            consumer_group: Kafka consumer group ID.
            topic: Primary Kafka topic.
            event_type: Type of health event.
            severity: Severity level.
            correlation_id: Optional correlation ID (auto-generated if None).
            rebalance_duration_ms: Rebalance duration in ms.
            partitions_assigned: Partitions assigned after rebalance.
            partitions_revoked: Partitions revoked during rebalance.
            error_message: Error message if applicable.
            error_type: Exception type name if applicable.
            hostname: Machine hostname.
            service_label: Service display name.

        Returns:
            A new ModelConsumerHealthEvent with computed fingerprint.
        """
        fingerprint = _compute_fingerprint(consumer_identity, event_type.value, topic)
        return cls(
            consumer_identity=consumer_identity,
            consumer_group=consumer_group,
            topic=topic,
            event_type=event_type,
            severity=severity,
            fingerprint=fingerprint,
            correlation_id=correlation_id or uuid4(),
            rebalance_duration_ms=rebalance_duration_ms,
            partitions_assigned=partitions_assigned,
            partitions_revoked=partitions_revoked,
            error_message=error_message,
            error_type=error_type,
            hostname=hostname,
            service_label=service_label,
        )


__all__ = [
    "ModelConsumerHealthEvent",
]

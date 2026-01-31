# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Observability Envelope Model.

This module defines the strict envelope model for all observability events.
The envelope contains common metadata fields that must be present on every
event consumed by the agent_actions observability consumer.

Design Decisions:
    - extra="forbid": Strict validation ensures envelope schema compliance
    - All fields are required or have explicit defaults
    - correlation_id is optional (some events may not have request context)

Thread Safety:
    ModelObservabilityEnvelope is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> envelope = ModelObservabilityEnvelope(
    ...     event_id=uuid4(),
    ...     event_time=datetime.now(UTC),
    ...     producer_id="agent-observability-postgres",
    ...     schema_version="1.0.0",
    ...     correlation_id=uuid4(),
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelObservabilityEnvelope(BaseModel):
    """Strict envelope for observability events.

    All events must have these fields. The envelope is validated with
    extra="forbid" to ensure no unexpected fields are present.

    Attributes:
        event_id: Unique identifier for this specific event instance.
        event_time: Timestamp when the event was produced.
        producer_id: Identifier of the service/component that produced the event.
        schema_version: Version of the event schema for evolution tracking.
        correlation_id: Optional request correlation ID for distributed tracing.

    Example:
        >>> envelope = ModelObservabilityEnvelope(
        ...     event_id=uuid4(),
        ...     event_time=datetime.now(UTC),
        ...     producer_id="claude-code-agent",
        ...     schema_version="1.0.0",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    event_id: UUID = Field(
        ...,
        description="Unique identifier for this specific event instance.",
    )
    event_time: datetime = Field(
        ...,
        description="Timestamp when the event was produced.",
    )
    producer_id: str = Field(  # ONEX_EXCLUDE: string_id - external service identifier
        ..., description="Identifier of the service/component that produced the event."
    )
    schema_version: str = Field(
        ...,
        description="Version of the event schema for evolution tracking.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Optional request correlation ID for distributed tracing.",
    )


__all__ = ["ModelObservabilityEnvelope"]

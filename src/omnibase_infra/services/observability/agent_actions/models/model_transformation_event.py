# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Transformation Event Model.

This module defines the model for agent transformation events consumed
from Kafka. Transformation events capture when an agent transforms into
a specialized form (e.g., polymorphic agent becoming api-architect).

Design Decisions:
    - extra="allow": Phase 1 flexibility - required fields typed, extras preserved
    - raw_payload: Optional field to preserve complete payload for schema tightening
    - created_at: Required for TTL cleanup job (Phase 2)

Idempotency:
    Table: agent_transformation_events
    Unique Key: id (UUID)
    Conflict Action: DO NOTHING (append-only audit log)

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> event = ModelTransformationEvent(
    ...     id=uuid4(),
    ...     correlation_id=uuid4(),
    ...     source_agent="polymorphic-agent",
    ...     target_agent="api-architect",
    ...     created_at=datetime.now(UTC),
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelTransformationEvent(BaseModel):
    """Agent transformation event model.

    Represents a transformation from one agent type to another, typically
    when a polymorphic agent specializes into a domain-specific agent.

    Attributes:
        id: Unique identifier for this event (idempotency key).
        correlation_id: Request correlation ID linking related events.
        source_agent: Name of the original agent before transformation.
        target_agent: Name of the agent after transformation.
        created_at: Timestamp when the transformation occurred (TTL key).
        trigger: Optional trigger that caused the transformation.
        context: Optional context information about the transformation.
        metadata: Optional additional metadata about the transformation.
        raw_payload: Optional complete raw payload for Phase 2 schema tightening.

    Example:
        >>> event = ModelTransformationEvent(
        ...     id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     source_agent="polymorphic-agent",
        ...     target_agent="debug-database",
        ...     created_at=datetime.now(UTC),
        ...     trigger="User requested database debugging",
        ... )
    """

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
    )

    # ---- Required Fields ----
    id: UUID = Field(
        ...,
        description="Unique identifier for this event (idempotency key).",
    )
    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID linking related events.",
    )
    source_agent: str = Field(
        ...,
        description="Name of the original agent before transformation.",
    )
    target_agent: str = Field(
        ...,
        description="Name of the agent after transformation.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the transformation occurred (TTL key).",
    )

    # ---- Optional Fields ----
    trigger: str | None = Field(
        default=None,
        description="Trigger that caused the transformation.",
    )
    context: str | None = Field(
        default=None,
        description="Context information about the transformation.",
    )
    metadata: dict[str, object] | None = Field(
        default=None,
        description="Additional metadata about the transformation.",
    )
    raw_payload: dict[str, object] | None = Field(
        default=None,
        description="Complete raw payload for Phase 2 schema tightening.",
    )


__all__ = ["ModelTransformationEvent"]

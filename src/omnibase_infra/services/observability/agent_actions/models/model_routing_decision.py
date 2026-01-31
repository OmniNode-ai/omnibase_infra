# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Routing Decision Model.

This module defines the model for agent routing decision events consumed
from Kafka. Routing decisions capture the selection logic when an agent
is chosen to handle a request, including confidence scores and alternatives.

Design Decisions:
    - extra="allow": Phase 1 flexibility - required fields typed, extras preserved
    - raw_payload: Optional field to preserve complete payload for schema tightening
    - created_at: Required for TTL cleanup job (Phase 2)

Idempotency:
    Table: agent_routing_decisions
    Unique Key: id (UUID)
    Conflict Action: DO NOTHING (append-only audit log)

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> decision = ModelRoutingDecision(
    ...     id=uuid4(),
    ...     correlation_id=uuid4(),
    ...     selected_agent="api-architect",
    ...     confidence_score=0.95,
    ...     created_at=datetime.now(UTC),
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelRoutingDecision(BaseModel):
    """Agent routing decision event model.

    Represents the routing decision made when selecting an agent to handle
    a request. Captures the selected agent, confidence score, and alternatives
    considered during routing.

    Attributes:
        id: Unique identifier for this decision (idempotency key).
        correlation_id: Request correlation ID linking related events.
        selected_agent: Name of the agent selected to handle the request.
        confidence_score: Confidence score (0.0-1.0) for the selection.
        created_at: Timestamp when the decision was made (TTL key).
        request_type: Optional type of request being routed.
        alternatives: Optional list of alternative agents considered.
        routing_reason: Optional explanation for the routing decision.
        domain: Optional domain classification for the request.
        metadata: Optional additional metadata about the decision.
        raw_payload: Optional complete raw payload for Phase 2 schema tightening.

    Example:
        >>> decision = ModelRoutingDecision(
        ...     id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     selected_agent="polymorphic-agent",
        ...     confidence_score=0.87,
        ...     created_at=datetime.now(UTC),
        ...     routing_reason="Matched domain pattern for infrastructure tasks",
        ... )
    """

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
    )

    # ---- Required Fields ----
    id: UUID = Field(
        ...,
        description="Unique identifier for this decision (idempotency key).",
    )
    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID linking related events.",
    )
    selected_agent: str = Field(
        ...,
        description="Name of the agent selected to handle the request.",
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score (0.0-1.0) for the selection.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the decision was made (TTL key).",
    )

    # ---- Optional Fields ----
    request_type: str | None = Field(
        default=None,
        description="Type of request being routed.",
    )
    alternatives: list[str] | None = Field(
        default=None,
        description="List of alternative agents considered.",
    )
    routing_reason: str | None = Field(
        default=None,
        description="Explanation for the routing decision.",
    )
    domain: str | None = Field(
        default=None,
        description="Domain classification for the request.",
    )
    metadata: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional metadata about the decision.",
    )
    raw_payload: dict[str, JsonType] | None = Field(
        default=None,
        description="Complete raw payload for Phase 2 schema tightening.",
    )


__all__ = ["ModelRoutingDecision"]

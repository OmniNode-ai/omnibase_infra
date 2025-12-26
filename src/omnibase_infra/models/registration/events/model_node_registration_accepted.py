# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Accepted Event Model.

This module provides ModelNodeRegistrationAccepted for the ONEX 2-way
registration pattern. Emitted by the Registration Orchestrator when it
accepts a node's registration request.

See Also:
    - docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md (C1 section)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistrationAccepted(BaseModel):
    """Event model for registration acceptance by the orchestrator.

    Emitted when the orchestrator decides to accept a node's registration.
    The node must acknowledge this acceptance within the ack_deadline to
    complete the registration handshake.

    This event triggers the reducer to emit intents for persisting the
    registration to Consul and PostgreSQL.

    Attributes:
        entity_id: The entity identifier (equals node_id for registration domain).
            Used as partition key for ordering guarantees.
        node_id: Unique identifier of the node being registered.
        correlation_id: Correlation ID for distributed tracing across the workflow.
        causation_id: Message ID of the event that triggered this decision.
        emitted_at: Timestamp when the orchestrator emitted this event (UTC).
        ack_deadline: Deadline by which the node must acknowledge registration.
            If not acknowledged by this time, NodeRegistrationAckTimedOut is emitted.

    Time Injection:
        The `emitted_at` field must be explicitly provided by the handler
        using its injected `now` parameter. Do NOT use datetime.now() directly.
        This ensures deterministic testing and consistent ordering across nodes.

    Example:
        >>> from datetime import UTC, datetime, timedelta
        >>> from uuid import uuid4
        >>> event = ModelNodeRegistrationAccepted(
        ...     entity_id=uuid4(),
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ...     ack_deadline=datetime(2025, 1, 15, 12, 0, 30, tzinfo=UTC),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Entity and node identification
    entity_id: UUID = Field(
        ...,
        description="Entity identifier (equals node_id for registration domain)",
    )
    node_id: UUID = Field(
        ...,
        description="Unique identifier of the node being registered",
    )

    # Tracing and causation
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    causation_id: UUID = Field(
        ...,
        description="Message ID of the event that triggered this decision",
    )

    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    emitted_at: datetime = Field(
        ...,
        description="Timestamp when the orchestrator emitted this event (UTC)",
    )

    # Registration-specific
    ack_deadline: datetime = Field(
        ...,
        description="Deadline by which the node must acknowledge registration (UTC)",
    )


__all__ = ["ModelNodeRegistrationAccepted"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Ack Received Event Model.

This module provides ModelNodeRegistrationAckReceived for the ONEX 2-way
registration pattern. Emitted by the Registration Orchestrator when it
receives acknowledgment from the node within the ack_deadline.

See Also:
    - docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md (C1 section)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistrationAckReceived(BaseModel):
    """Event model for registration acknowledgment received.

    Emitted when the orchestrator processes a NodeRegistrationAcked command
    from the node within the ack_deadline. This completes the handshake and
    starts the liveness monitoring phase.

    After this event, the node enters a monitored state where it must send
    heartbeats before the liveness_deadline expires.

    Attributes:
        entity_id: The entity identifier (equals node_id for registration domain).
            Used as partition key for ordering guarantees.
        node_id: Unique identifier of the node that acknowledged registration.
        correlation_id: Correlation ID for distributed tracing across the workflow.
        causation_id: Message ID of the NodeRegistrationAcked command.
        emitted_at: Timestamp when the orchestrator emitted this event (UTC).
        liveness_deadline: Deadline for the next heartbeat from the node.
            If no heartbeat is received by this time, NodeLivenessExpired is emitted.

    Time Injection:
        The `emitted_at` field must be explicitly provided by the handler
        using its injected `now` parameter. Do NOT use datetime.now() directly.
        This ensures deterministic testing and consistent ordering across nodes.

    Example:
        >>> from datetime import UTC, datetime, timedelta
        >>> from uuid import uuid4
        >>> event = ModelNodeRegistrationAckReceived(
        ...     entity_id=uuid4(),
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ...     liveness_deadline=datetime(2025, 1, 15, 12, 1, 0, tzinfo=UTC),
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
        description="Unique identifier of the node that acknowledged registration",
    )

    # Tracing and causation
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    causation_id: UUID = Field(
        ...,
        description="Message ID of the NodeRegistrationAcked command",
    )

    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    emitted_at: datetime = Field(
        ...,
        description="Timestamp when the orchestrator emitted this event (UTC)",
    )

    # Liveness-specific
    liveness_deadline: datetime = Field(
        ...,
        description="Deadline for the next heartbeat from the node (UTC)",
    )


__all__ = ["ModelNodeRegistrationAckReceived"]

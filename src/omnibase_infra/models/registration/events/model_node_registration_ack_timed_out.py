# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Ack Timeout Event Model.

This module provides the event model emitted when a node fails to acknowledge
its registration within the configured deadline.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistrationAckTimedOut(BaseModel):
    """Event emitted when a node's registration ack deadline has passed.

    This event is produced by the registration orchestrator during RuntimeTick
    processing when it detects that a node in ACCEPTED or AWAITING_ACK state
    has exceeded its ack_deadline.

    Event Semantics (per C2 Durable Timeout Handling):
        - Emitted once per entity per timeout occurrence
        - Uses emission markers in projection for deduplication
        - Links to triggering tick via causation_id
        - Survives orchestrator restarts (deadline stored in projection)

    FSM Impact:
        This event triggers the FSM transition:
        AWAITING_ACK -> ACK_TIMED_OUT

    Attributes:
        entity_id: The registration entity identifier (same as node_id)
        node_id: The node UUID that failed to acknowledge
        correlation_id: Correlation ID for distributed tracing
        causation_id: UUID of the RuntimeTick that triggered this event
        emitted_at: When the timeout was detected (from tick.now)
        deadline_at: The original ack deadline that was exceeded

    Example:
        >>> from datetime import datetime, UTC, timedelta
        >>> from uuid import uuid4
        >>> event = ModelNodeRegistrationAckTimedOut(
        ...     entity_id=uuid4(),
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=datetime.now(UTC),
        ...     deadline_at=datetime.now(UTC) - timedelta(minutes=5),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Entity identification
    entity_id: UUID = Field(
        ...,
        description="The registration entity identifier (same as node_id)",
    )
    node_id: UUID = Field(
        ...,
        description="The node UUID that failed to acknowledge registration",
    )

    # Tracing
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    causation_id: UUID = Field(
        ...,
        description="UUID of the RuntimeTick that triggered this timeout event",
    )

    # Timing - MUST be explicitly injected (no default_factory for testability)
    emitted_at: datetime = Field(
        ...,
        description="When the timeout was detected (from RuntimeTick.now)",
    )
    deadline_at: datetime = Field(
        ...,
        description="The original ack deadline that was exceeded",
    )


__all__: list[str] = ["ModelNodeRegistrationAckTimedOut"]

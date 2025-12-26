# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Liveness Expired Event Model.

This module provides the event model emitted when an active node fails to
send heartbeats within the configured liveness deadline.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeLivenessExpired(BaseModel):
    """Event emitted when an active node's liveness deadline has passed.

    This event is produced by the registration orchestrator during RuntimeTick
    processing when it detects that an ACTIVE node has exceeded its
    liveness_deadline without sending a heartbeat.

    Event Semantics (per C2 Durable Timeout Handling):
        - Emitted once per entity per timeout occurrence
        - Uses emission markers in projection for deduplication
        - Links to triggering tick via causation_id
        - Survives orchestrator restarts (deadline stored in projection)

    FSM Impact:
        This event triggers the FSM transition:
        ACTIVE -> LIVENESS_EXPIRED (terminal state)

    Attributes:
        entity_id: The registration entity identifier (same as node_id)
        node_id: The node UUID that failed liveness check
        correlation_id: Correlation ID for distributed tracing
        causation_id: UUID of the RuntimeTick that triggered this event
        emitted_at: When the liveness expiry was detected (from tick.now)
        last_heartbeat_at: The timestamp of the last received heartbeat

    Time Injection:
        The `emitted_at` field must be explicitly provided by the handler
        using its injected `now` parameter. Do NOT use datetime.now() directly.
        This ensures deterministic testing and consistent ordering across nodes.

    Example:
        >>> from datetime import datetime, UTC, timedelta
        >>> from uuid import uuid4
        >>> # Use explicit timestamps (time injection pattern) - not datetime.now()
        >>> now = datetime(2025, 1, 15, 12, 10, 0, tzinfo=UTC)
        >>> event = ModelNodeLivenessExpired(
        ...     entity_id=uuid4(),
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=now,
        ...     last_heartbeat_at=now - timedelta(minutes=10),
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
        description="The node UUID that failed liveness check",
    )

    # Tracing
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    causation_id: UUID = Field(
        ...,
        description="UUID of the RuntimeTick that triggered this liveness expired event",
    )

    # Timing - MUST be explicitly injected (no default_factory for testability)
    emitted_at: datetime = Field(
        ...,
        description="When the liveness expiry was detected (from RuntimeTick.now)",
    )
    last_heartbeat_at: datetime | None = Field(
        ...,
        description="The timestamp of the last received heartbeat (None if never received)",
    )


__all__: list[str] = ["ModelNodeLivenessExpired"]

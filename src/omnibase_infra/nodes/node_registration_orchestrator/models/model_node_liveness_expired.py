# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Liveness expired event model for the registration orchestrator.

Emitted when an active node misses its liveness heartbeat deadline.
This is a timeout decision event that triggers state transition to LIVENESS_EXPIRED.

Timestamp Accuracy Verification (2025-12-25):
    All timestamps in this event model are verified accurate:

    - detected_at: Fresh UTC timestamp from RuntimeTick.now at detection time.
      Source: RuntimeScheduler.emit_tick() -> datetime.now(UTC) -> process_timeouts().
      NOT cached or stale.

    - liveness_deadline: The deadline that was missed, sourced from the
      registration projection (calculated as last_heartbeat_at + liveness_window
      when heartbeat was processed).

    - last_heartbeat_at: Optional timestamp of last heartbeat, sourced from
      projection.last_heartbeat_at (set from heartbeat event timestamp).

    UTC timezone handling is enforced at the RuntimeScheduler level using
    datetime.now(UTC). See test_timeout_emitter.py for comprehensive tests.

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-888 (C1): Registration Orchestrator
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeLivenessExpired(BaseModel):
    """Event emitted when an active node misses liveness heartbeat deadline.

    This is a timeout decision event emitted by the orchestrator when:
    - liveness_deadline has passed
    - Node state is ACTIVE
    - No previous expiry event was emitted (liveness_timeout_emitted_at was None)

    The orchestrator emits this event during RuntimeTick processing when
    scanning projections for missed heartbeat deadlines. The reducer will
    transition the node to LIVENESS_EXPIRED state (terminal) upon receiving
    this event.

    Topic Pattern:
        {env}.{namespace}.onex.evt.node-liveness-expired.v1

    Attributes:
        node_id: UUID of the node whose liveness expired.
        liveness_deadline: The deadline that was missed.
        detected_at: When the expiry was detected (from RuntimeTick.now).
        last_heartbeat_at: When the last heartbeat was received (if tracked).
        correlation_id: Correlation ID from the RuntimeTick that triggered detection.
        causation_id: Message ID that caused this event (RuntimeTick.tick_id).

    Example:
        >>> from datetime import datetime, UTC, timedelta
        >>> from uuid import uuid4
        >>> now = datetime.now(UTC)
        >>> event = ModelNodeLivenessExpired(
        ...     node_id=uuid4(),
        ...     liveness_deadline=now - timedelta(seconds=30),
        ...     detected_at=now,
        ...     last_heartbeat_at=now - timedelta(minutes=2),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    node_id: UUID = Field(
        ...,
        description="UUID of the node whose liveness expired",
    )
    liveness_deadline: datetime = Field(
        ...,
        description="The deadline that was missed",
    )
    detected_at: datetime = Field(
        ...,
        description="When the expiry was detected (from RuntimeTick.now)",
    )
    last_heartbeat_at: datetime | None = Field(
        default=None,
        description="When last heartbeat was received (if tracked)",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID from the RuntimeTick that triggered detection",
    )
    causation_id: UUID = Field(
        ...,
        description="Message ID that caused this event (RuntimeTick.tick_id)",
    )


__all__ = ["ModelNodeLivenessExpired"]

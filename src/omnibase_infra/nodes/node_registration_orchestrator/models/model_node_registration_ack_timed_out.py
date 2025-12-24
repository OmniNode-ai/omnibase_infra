# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Ack timeout event model for the registration orchestrator.

Emitted when a node fails to acknowledge registration within the ack_deadline.
This is a timeout decision event that triggers state transition to ACK_TIMED_OUT.

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-888 (C1): Registration Orchestrator
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumRegistrationState


class ModelNodeRegistrationAckTimedOut(BaseModel):
    """Event emitted when a node fails to acknowledge registration within deadline.

    This is a timeout decision event emitted by the orchestrator when:
    - ack_deadline has passed
    - Node state is ACCEPTED or AWAITING_ACK (requires_ack() is True)
    - No previous timeout event was emitted (ack_timeout_emitted_at was None)

    The orchestrator emits this event during RuntimeTick processing when
    scanning projections for missed deadlines. The reducer will transition
    the node to ACK_TIMED_OUT state upon receiving this event.

    Topic Pattern:
        {env}.{namespace}.onex.evt.node-registration-ack-timed-out.v1

    Attributes:
        node_id: UUID of the node that timed out.
        ack_deadline: The deadline that was missed.
        detected_at: When the timeout was detected (from RuntimeTick.now).
        previous_state: State before timeout (ACCEPTED or AWAITING_ACK).
        correlation_id: Correlation ID from the RuntimeTick that triggered detection.
        causation_id: Message ID that caused this event (RuntimeTick.tick_id).

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> event = ModelNodeRegistrationAckTimedOut(
        ...     node_id=uuid4(),
        ...     ack_deadline=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        ...     detected_at=datetime(2025, 1, 1, 12, 5, 0, tzinfo=UTC),
        ...     previous_state=EnumRegistrationState.AWAITING_ACK,
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
        description="UUID of the node that timed out",
    )
    ack_deadline: datetime = Field(
        ...,
        description="The deadline that was missed",
    )
    detected_at: datetime = Field(
        ...,
        description="When the timeout was detected (from RuntimeTick.now)",
    )
    previous_state: EnumRegistrationState = Field(
        ...,
        description="State before timeout (ACCEPTED or AWAITING_ACK)",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID from the RuntimeTick that triggered detection",
    )
    causation_id: UUID = Field(
        ...,
        description="Message ID that caused this event (RuntimeTick.tick_id)",
    )


__all__ = ["ModelNodeRegistrationAckTimedOut"]

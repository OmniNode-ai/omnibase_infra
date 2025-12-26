# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Acknowledged Command Model.

This module provides ModelNodeRegistrationAcked, a command sent by nodes
to acknowledge their registration in the ONEX 2-way registration pattern.

Command vs Event Distinction:
    - COMMANDS are imperative requests from external sources
    - EVENTS are facts about things that have happened
    - NodeRegistrationAcked is a COMMAND because it's the node requesting
      acknowledgment of its registration (imperative)
    - NodeRegistrationAckReceived is the EVENT emitted when the orchestrator
      processes this command successfully (fact)

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistrationAcked(BaseModel):
    """Command: Node acknowledges its registration.

    This is a COMMAND, not an event. Commands are imperative requests
    from external sources (the node itself in this case) that orchestrators
    process to make decisions and emit events.

    The orchestrator receives this command and, if valid:
    1. Emits NodeRegistrationAckReceived event
    2. Emits NodeBecameActive event (if transitioning to active state)

    Validity Conditions:
        - Node must be in AWAITING_ACK state (queried from projection)
        - If node is already ACTIVE, this is a duplicate ack (no-op)
        - If node is in terminal state, this ack is too late (rejected)

    Attributes:
        command_id: Unique identifier for this command instance.
        node_id: The UUID of the node sending the acknowledgment.
        correlation_id: Correlation ID for distributed tracing, linking
            this command to the original registration flow.
        timestamp: When the node sent the acknowledgment.

    Time Injection:
        The `timestamp` field must be explicitly provided by the caller
        using an injected `now` parameter. Do NOT use datetime.now() directly.
        This ensures deterministic testing and consistent ordering across nodes.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> ack = ModelNodeRegistrationAcked(
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ... )
        >>> assert ack.command_id is not None
        >>> assert ack.timestamp.tzinfo is not None

    See Also:
        - ModelNodeRegistrationAckReceived: Event emitted when ack is processed
        - ModelNodeBecameActive: Event emitted when node becomes active
        - HandlerNodeRegistrationAcked: Handler that processes this command
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Command identification
    command_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this command instance.",
    )

    # Node identification
    node_id: UUID = Field(
        ...,
        description="The UUID of the node sending the acknowledgment.",
    )

    # Tracing
    correlation_id: UUID = Field(
        ...,
        description=(
            "Correlation ID for distributed tracing, linking this command "
            "to the original registration flow."
        ),
    )

    # Timestamp - MUST be explicitly injected (no default_factory for testability)
    timestamp: datetime = Field(
        ...,
        description="When the node sent the acknowledgment.",
    )


__all__: list[str] = ["ModelNodeRegistrationAcked"]

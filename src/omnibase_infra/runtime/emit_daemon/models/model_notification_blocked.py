# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Notification Blocked Event Model.

This model represents a notification event emitted when an agent is blocked
waiting for human input during ticket work execution.

Design:
    - Event payload for `notification.blocked` event type
    - Published via emit daemon to Kafka
    - Consumed by notification consumer and routed to Slack

Example Payload:
    ```json
    {
        "ticket_identifier": "OMN-1234",
        "reason": "Waiting for specification approval",
        "details": ["Phase: spec", "Gate: approve spec"],
        "repo": "omniclaude",
        "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```

Related Tickets:
    - OMN-1831: Implement event-driven Slack notifications via runtime
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelNotificationBlocked(BaseModel):
    """Event model for agent blocked notifications.

    This model represents the payload structure for notification.blocked events
    emitted when an agent needs human input to proceed with ticket work.

    Attributes:
        ticket_identifier: Linear ticket identifier (e.g., "OMN-1234").
        reason: Human-readable reason for being blocked.
        details: List of additional context items providing more information.
        repo: Repository name where work is happening.
        session_id: Claude Code session UUID for correlation.
        correlation_id: Optional UUID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> blocked = ModelNotificationBlocked(
        ...     ticket_identifier="OMN-1234",
        ...     reason="Waiting for specification approval",
        ...     details=["Phase: spec", "Gate: approve spec"],
        ...     repo="omniclaude",
        ...     session_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_identifier: str = Field(
        ...,
        description="Linear ticket identifier (e.g., OMN-1234)",
        min_length=1,
    )
    reason: str = Field(
        ...,
        description="Human-readable reason for being blocked",
        min_length=1,
    )
    details: list[str] = Field(
        default_factory=list,
        description="List of additional context items",
    )
    repo: str = Field(
        ...,
        description="Repository name where work is happening",
        min_length=1,
    )
    session_id: UUID = Field(
        ...,
        description="Claude Code session UUID for correlation",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for distributed tracing",
    )


__all__ = ["ModelNotificationBlocked"]

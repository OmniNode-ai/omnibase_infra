# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Notification Completed Event Model.

This model represents a notification event emitted when ticket work is
completed successfully.

Design:
    - Event payload for `notification.completed` event type
    - Published via emit daemon to Kafka
    - Consumed by notification consumer and routed to Slack

Example Payload:
    ```json
    {
        "ticket_identifier": "OMN-1234",
        "summary": "Implemented dark mode feature",
        "repo": "omniclaude",
        "pr_url": "https://github.com/org/repo/pull/123",
        "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```

Related Tickets:
    - OMN-1831: Implement event-driven Slack notifications via runtime
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelNotificationCompleted(BaseModel):
    """Event model for ticket work completion notifications.

    This model represents the payload structure for notification.completed
    events emitted when ticket work finishes successfully.

    Attributes:
        ticket_identifier: Linear ticket identifier (e.g., "OMN-1234").
        summary: Brief description of what was accomplished.
        repo: Repository name where work was done.
        pr_url: Optional pull request URL.
        session_id: Claude Code session UUID for correlation.
        correlation_id: Optional UUID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> completed = ModelNotificationCompleted(
        ...     ticket_identifier="OMN-1234",
        ...     summary="Implemented dark mode feature",
        ...     repo="omniclaude",
        ...     pr_url="https://github.com/org/repo/pull/123",
        ...     session_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_identifier: str = Field(
        ...,
        description="Linear ticket identifier (e.g., OMN-1234)",
        min_length=1,
    )
    summary: str = Field(
        ...,
        description="Brief description of what was accomplished",
        min_length=1,
    )
    repo: str = Field(
        ...,
        description="Repository name where work was done",
        min_length=1,
    )
    pr_url: str | None = Field(
        default=None,
        description="Optional pull request URL",
    )
    session_id: UUID = Field(
        ...,
        description="Claude Code session UUID for correlation",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for distributed tracing",
    )


__all__ = ["ModelNotificationCompleted"]

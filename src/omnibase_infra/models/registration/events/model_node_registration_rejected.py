# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Rejected Event Model.

This module provides ModelNodeRegistrationRejected for the ONEX 2-way
registration pattern. Emitted by the Registration Orchestrator when it
rejects a node's registration request.

See Also:
    - docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md (C1 section)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistrationRejected(BaseModel):
    """Event model for registration rejection by the orchestrator.

    Emitted when the orchestrator decides to reject a node's registration.
    This is a terminal state for the registration attempt - the node must
    re-introspect to attempt registration again.

    Common rejection reasons include:
    - Node version incompatibility
    - Capability requirements not met
    - Rate limiting exceeded
    - Duplicate registration attempt
    - Policy violation

    Attributes:
        entity_id: The entity identifier (equals node_id for registration domain).
            Used as partition key for ordering guarantees.
        node_id: Unique identifier of the node being rejected.
        correlation_id: Correlation ID for distributed tracing across the workflow.
        causation_id: Message ID of the event that triggered this decision.
        emitted_at: Timestamp when the orchestrator emitted this event (UTC).
        rejection_reason: Human-readable explanation for the rejection.
            Should be safe to expose to the node (no internal details).

    Time Injection:
        The `emitted_at` field must be explicitly provided by the handler
        using its injected `now` parameter. Do NOT use datetime.now() directly.
        This ensures deterministic testing and consistent ordering across nodes.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> event = ModelNodeRegistrationRejected(
        ...     entity_id=uuid4(),
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ...     rejection_reason="Node version 0.9.0 is below minimum required 1.0.0",
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
        description="Unique identifier of the node being rejected",
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

    # Rejection-specific
    rejection_reason: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Human-readable explanation for the rejection",
    )


__all__ = ["ModelNodeRegistrationRejected"]

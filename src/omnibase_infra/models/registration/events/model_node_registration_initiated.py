# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Initiated Event Model.

This module provides ModelNodeRegistrationInitiated for the ONEX 2-way
registration pattern. Emitted by the Registration Orchestrator when it
receives a NodeIntrospected event to represent the start of a registration
attempt.

IMPORTANT: Timestamp fields (emitted_at) have NO default_factory (no datetime.now()).
This is intentional per ONEX architecture - orchestrators use injected `now` parameter
from RuntimeTick or dispatch context. This ensures:
    - Deterministic testing (fixed time in tests)
    - Consistent ordering (no clock skew between nodes)
    - Explicit time injection (no hidden time dependencies)

See Also:
    - docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md (C1 section)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistrationInitiated(BaseModel):
    """Event model for registration initiation by the orchestrator.

    Emitted when the orchestrator receives a NodeIntrospected event and
    begins processing a registration attempt. This event represents the
    start of the registration workflow without implying command-driven
    semantics.

    This is the first decision event in the registration workflow, linking
    the NodeIntrospected trigger to subsequent orchestrator decisions.

    Attributes:
        entity_id: The entity identifier (equals node_id for registration domain).
            Used as partition key for ordering guarantees.
        node_id: Unique identifier of the node being registered.
        correlation_id: Correlation ID for distributed tracing across the workflow.
        causation_id: Message ID of the triggering NodeIntrospected event.
        emitted_at: Timestamp when the orchestrator emitted this event (UTC).
        registration_attempt_id: Unique identifier for this registration attempt.
            Enables tracking multiple registration attempts for the same node.

    Time Injection:
        The `emitted_at` field must be explicitly provided by the handler
        using its injected `now` parameter. Do NOT use datetime.now() directly.
        This ensures deterministic testing and consistent ordering across nodes.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> event = ModelNodeRegistrationInitiated(
        ...     entity_id=uuid4(),
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ...     registration_attempt_id=uuid4(),
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
        description="Message ID of the triggering NodeIntrospected event",
    )

    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    emitted_at: datetime = Field(
        ...,
        description="Timestamp when the orchestrator emitted this event (UTC)",
    )

    # Registration-specific
    registration_attempt_id: UUID = Field(
        ...,
        description="Unique identifier for this registration attempt",
    )


__all__ = ["ModelNodeRegistrationInitiated"]

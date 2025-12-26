# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Became Active Event Model.

This module provides ModelNodeBecameActive for the ONEX 2-way registration
pattern. Emitted by the Registration Orchestrator when a node transitions
to the active state.

See Also:
    - docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md (C1 section)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)


class ModelNodeBecameActive(BaseModel):
    """Event model for node activation.

    Emitted when the orchestrator transitions a node to the active state.
    This typically occurs after the node has successfully acknowledged
    registration and is ready to participate in the cluster.

    The capabilities field captures the node's advertised capabilities
    at the time of activation, enabling routing and discovery decisions.

    Attributes:
        entity_id: The entity identifier (equals node_id for registration domain).
            Used as partition key for ordering guarantees.
        node_id: Unique identifier of the activated node.
        correlation_id: Correlation ID for distributed tracing across the workflow.
        causation_id: Message ID of the event that triggered activation.
        emitted_at: Timestamp when the orchestrator emitted this event (UTC).
        capabilities: The node's capabilities at activation time.
            Used for routing and service discovery decisions.

    Time Injection:
        The `emitted_at` field must be explicitly provided by the handler
        using its injected `now` parameter. Do NOT use datetime.now() directly.
        This ensures deterministic testing and consistent ordering across nodes.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> from omnibase_infra.models.registration import ModelNodeCapabilities
        >>> event = ModelNodeBecameActive(
        ...     entity_id=uuid4(),
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ...     capabilities=ModelNodeCapabilities(postgres=True, read=True),
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
        description="Unique identifier of the activated node",
    )

    # Tracing and causation
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    causation_id: UUID = Field(
        ...,
        description="Message ID of the event that triggered activation",
    )

    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    emitted_at: datetime = Field(
        ...,
        description="Timestamp when the orchestrator emitted this event (UTC)",
    )

    # Activation-specific
    capabilities: ModelNodeCapabilities = Field(
        ...,
        description="The node's capabilities at activation time",
    )


__all__ = ["ModelNodeBecameActive"]

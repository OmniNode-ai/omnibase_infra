# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Orchestrator Context Model for Handler Execution.

This module provides a simplified context model for orchestrator handlers.
While ModelDispatchContext is used by the dispatch engine for routing,
ModelOrchestratorContext is passed directly to handlers during execution.

Design Note:
    This context carries the essential information handlers need:
    - now: Injected current time for timeout decisions
    - correlation_id: For distributed tracing
    - trace_id: Optional additional tracing identifier

    The `now` field is the single source of truth for time within a handler
    execution, matching the time injection pattern from RuntimeTick (B6)
    and ModelDispatchContext.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-948: Context Time Injection Pattern
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelOrchestratorContext(BaseModel):
    """Context passed to orchestrator handlers during execution.

    Provides handlers with the essential context needed for execution:
    - Time injection for timeout and deadline decisions
    - Correlation ID for distributed tracing
    - Optional trace ID for additional tracing context

    Thread Safety:
        This model is immutable (frozen=True) after creation,
        making it thread-safe for concurrent read access.

    Attributes:
        now: Injected current time for handler decisions. This is the
             authoritative time source - handlers must use this instead
             of calling datetime.now() directly.
        correlation_id: Correlation ID for distributed tracing. All events
                       emitted by handlers should include this ID.
        trace_id: Optional trace identifier for distributed tracing systems.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> context = ModelOrchestratorContext(
        ...     now=datetime.now(UTC),
        ...     correlation_id=uuid4(),
        ...     trace_id=uuid4(),
        ... )
        >>> # Handler uses context.now for timeout checks
        >>> if projection.has_ack_deadline_passed(context.now):
        ...     emit_timeout_event(context.correlation_id)
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Time injection
    now: datetime = Field(
        ...,
        description=(
            "Injected current time for handler decisions. "
            "This is the authoritative time source within the handler execution."
        ),
    )

    # Tracing
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing.",
    )
    trace_id: UUID | None = Field(
        default=None,
        description="Optional trace identifier for distributed tracing systems.",
    )


__all__: list[str] = ["ModelOrchestratorContext"]

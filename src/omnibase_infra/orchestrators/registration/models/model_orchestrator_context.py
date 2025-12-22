# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Orchestrator Context Model.

This module defines the ModelOrchestratorContext which is passed to orchestrator
handlers. The context provides injected time and correlation tracking for
orchestrator decision-making.

Key Design Decisions:
    - `now`: Injected by runtime from RuntimeTick or dispatch time. This is the
      authoritative wall-clock time for all orchestrator decisions including
      deadline evaluation and timeout handling.
    - `correlation_id`: Propagated from incoming envelope for distributed tracing.
      All events emitted by the orchestrator must share this correlation_id.
    - `trace_id`: Optional distributed tracing identifier.

Thread Safety:
    ModelOrchestratorContext is frozen (immutable) for thread safety. Once created,
    an instance cannot be modified.

Usage:
    Orchestrator handlers receive this context and use `context.now` for all
    time-based decisions. They must NEVER call datetime.now() directly.

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> context = ModelOrchestratorContext(
    ...     now=datetime.now(UTC),
    ...     correlation_id=uuid4(),
    ... )
    >>> # Handler uses context.now for deadline checks
    >>> if projection.ack_deadline < context.now:
    ...     emit_ack_timeout_event()

Related:
    - ModelRuntimeTick: Source of `now` during tick-based execution
    - ModelDispatchContext: Lower-level dispatch context (used by runtime)
    - OMN-948: Time injection pattern specification
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelOrchestratorContext(BaseModel):
    """Context model for orchestrator handler execution.

    Contains injected time and correlation metadata for orchestrator
    decision-making. Orchestrators use this context for:
    - Deadline evaluation (comparing now against ack/liveness deadlines)
    - Timeout detection (during RuntimeTick processing)
    - Correlation propagation (all emitted events share correlation_id)

    Attributes:
        now: Injected wall-clock time (UTC, timezone-aware). The single source
             of truth for time-based decisions. NEVER call datetime.now() in
             orchestrator code - always use this field.
        correlation_id: Request correlation ID for distributed tracing. All
                       events emitted by the orchestrator must propagate this.
        trace_id: Optional distributed tracing identifier for external tracing
                  systems (e.g., OpenTelemetry).

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> ctx = ModelOrchestratorContext(
        ...     now=datetime.now(UTC),
        ...     correlation_id=uuid4(),
        ...     trace_id=uuid4(),
        ... )
        >>> assert ctx.now.tzinfo is not None  # Always timezone-aware
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for thread safety
        extra="forbid",  # Strict validation - no extra fields allowed
        from_attributes=True,
    )

    # Time injection - required for all orchestrator operations
    now: datetime = Field(
        ...,
        description=(
            "Injected wall-clock time (UTC, timezone-aware). The single source "
            "of truth for all orchestrator time-based decisions including deadline "
            "evaluation and timeout handling. NEVER call datetime.now() directly - "
            "always use this field."
        ),
    )

    # Correlation tracking - required for event propagation
    correlation_id: UUID = Field(
        ...,
        description=(
            "Request correlation ID for distributed tracing. All events emitted "
            "by the orchestrator must propagate this ID for workflow traceability."
        ),
    )

    # Optional distributed tracing
    trace_id: UUID | None = Field(
        default=None,
        description=(
            "Optional distributed tracing identifier for external tracing systems "
            "(e.g., OpenTelemetry, Jaeger). Propagated to emitted events if present."
        ),
    )


__all__: list[str] = ["ModelOrchestratorContext"]

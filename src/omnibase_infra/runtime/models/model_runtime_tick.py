# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Tick Event Model.

This module defines the ModelRuntimeTick infrastructure event model, which is emitted
by the runtime scheduler at configured intervals. This is an INFRASTRUCTURE concern,
not a domain event.

Infrastructure vs Domain Separation:
    - RuntimeTick is emitted by the runtime scheduler (infrastructure concern)
    - Orchestrators SUBSCRIBE to RuntimeTick and derive timeout decisions (domain concern)
    - The `now` field is the single source of truth for current time across the system
    - This field aligns with ModelOrchestratorContext.now and ModelEffectContext.now

Specification Reference:
    This model implements ticket B6 (Runtime Scheduler) from the ONEX Runtime Registration
    Ticket Plan. See docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md for full spec.

Key Design Decisions:
    - `now`: The authoritative wall-clock time for this tick. Orchestrators use this
      for timeout calculations. Reducers MUST NOT use this field.
    - `sequence_number`: Monotonically increasing counter per runtime instance.
      Enables restart-safety and ordering guarantees within a single runtime.
    - `tick_id`: Unique identifier for this specific tick emission. Used for
      idempotency and distributed tracing.
    - `scheduler_id`: Identifies which runtime scheduler instance emitted this tick.
      Critical for multi-runtime deployments where each runtime emits its own ticks.

Usage:
    Runtime schedulers emit this event on a configurable cadence:
    - Default: 1000ms (1 second)
    - Min: 100ms (prevents excessive CPU usage)
    - Max: 60000ms (ensures timely timeout detection)

    Configuration via: ONEX_RUNTIME_TICK_INTERVAL_MS environment variable

Example:
    >>> from datetime import datetime, timezone
    >>> from uuid import uuid4
    >>> # Use explicit timestamps (time injection pattern) - not datetime.now()
    >>> tick_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    >>> tick = ModelRuntimeTick(
    ...     now=tick_time,
    ...     tick_id=uuid4(),
    ...     sequence_number=42,
    ...     scheduled_at=tick_time,
    ...     correlation_id=uuid4(),
    ...     scheduler_id="runtime-instance-001",
    ...     tick_interval_ms=1000,
    ... )
    >>> print(tick.sequence_number)
    42
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeTick(BaseModel):
    """Infrastructure event emitted by runtime scheduler at configured intervals.

    Orchestrators subscribe to this event and derive timeout decisions from it.
    The `now` field is the single source of truth for current time across the system,
    matching the time injection pattern used in ModelOrchestratorContext and
    ModelEffectContext (see OMN-948).

    This is an INFRASTRUCTURE event, not a domain event. The distinction is:
    - Infrastructure: RuntimeTick emission (scheduler responsibility)
    - Domain: Timeout decisions derived from ticks (orchestrator responsibility)

    Attributes:
        now: The authoritative wall-clock time for this tick. UTC, timezone-aware.
             Orchestrators use this for deadline/timeout calculations.
             Effects may use this for retry/metrics timing.
             Reducers MUST ignore this field and never depend on it.
        tick_id: Unique identifier for this tick emission. Used for idempotency
                 guards and distributed tracing.
        sequence_number: Monotonically increasing counter per runtime instance.
                        Provides ordering guarantees and restart-safety detection.
                        Resets to 0 on runtime restart.
        scheduled_at: When this tick was scheduled to occur. May differ from `now`
                     if there was processing delay or clock skew.
        correlation_id: Correlation ID for distributed tracing. All downstream
                       operations triggered by this tick share this ID.
        scheduler_id: Identifies the runtime scheduler instance that emitted this tick.
                     Critical for multi-runtime deployments to avoid tick duplication.
        tick_interval_ms: The configured tick interval in milliseconds. Informational
                         field indicating the scheduler's cadence configuration.

    Thread Safety:
        This model is frozen (immutable) for thread safety. Once created,
        a RuntimeTick instance cannot be modified.

    Restart Safety:
        The sequence_number field enables detection of runtime restarts.
        If sequence_number < expected, a restart has occurred. Orchestrators
        should handle this gracefully by resetting their timeout calculations.

    Example:
        >>> from datetime import datetime, timezone
        >>> from uuid import uuid4
        >>> # Use explicit timestamps (time injection pattern) - not datetime.now()
        >>> tick_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        >>> tick = ModelRuntimeTick(
        ...     now=tick_time,
        ...     tick_id=uuid4(),
        ...     sequence_number=1,
        ...     scheduled_at=tick_time,
        ...     correlation_id=uuid4(),
        ...     scheduler_id="runtime-001",
        ...     tick_interval_ms=1000,
        ... )
        >>> assert tick.tick_interval_ms == 1000
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for thread safety
        extra="forbid",  # Strict validation - no extra fields allowed
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    # Core time field - matches handler context injection pattern (OMN-948)
    now: datetime = Field(
        ...,
        description=(
            "The authoritative wall-clock time for this tick. UTC, timezone-aware. "
            "Orchestrators use this for timeout calculations. Reducers MUST ignore."
        ),
    )

    # Tick identification
    tick_id: UUID = Field(
        ...,
        description=(
            "Unique identifier for this specific tick emission. "
            "Used for idempotency and distributed tracing."
        ),
    )

    # Ordering and restart-safety
    sequence_number: int = Field(
        ...,
        ge=0,
        description=(
            "Monotonically increasing counter per runtime instance. "
            "Provides ordering guarantees and restart-safety detection. "
            "Resets to 0 on runtime restart."
        ),
    )

    # Scheduling metadata
    scheduled_at: datetime = Field(
        ...,
        description=(
            "When this tick was scheduled to occur. May differ from `now` "
            "if there was processing delay or clock skew."
        ),
    )

    # Tracing
    correlation_id: UUID = Field(
        ...,
        description=(
            "Correlation ID for distributed tracing. All downstream "
            "operations triggered by this tick share this ID."
        ),
    )

    # Scheduler identification
    scheduler_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Identifies which runtime scheduler instance emitted this tick. "
            "Critical for multi-runtime deployments."
        ),
    )

    # Configuration metadata
    tick_interval_ms: int = Field(
        ...,
        ge=100,
        le=60000,
        description=(
            "Configured tick interval in milliseconds. "
            "Valid range: 100ms (min) to 60000ms (max)."
        ),
    )


__all__: list[str] = ["ModelRuntimeTick"]

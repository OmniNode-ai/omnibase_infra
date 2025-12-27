# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Runtime Scheduler Metrics Model.

This module provides the Pydantic model for tracking runtime scheduler performance
and operational metrics. Used for monitoring tick emission reliability, timing
performance, and scheduler health.

Metrics Categories:
    - Tick Counts: Total ticks emitted and failed
    - Timing Metrics: Duration tracking for individual ticks
    - Sequence Tracking: For restart-safety and exactly-once semantics
    - Circuit Breaker: Failure tolerance state
    - Uptime: Scheduler operational duration

Thread Safety:
    This model is immutable (frozen=True) and safe to share across threads.
    Create new instances for updated metrics using model_copy(update={...}).

Example:
    >>> from datetime import datetime, timezone
    >>> # Use explicit timestamps for deterministic behavior
    >>> last_tick = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    >>> metrics = ModelRuntimeSchedulerMetrics(
    ...     scheduler_id="scheduler-001",
    ...     ticks_emitted=100,
    ...     last_tick_at=last_tick,
    ... )
    >>> print(metrics.tick_success_rate())
    1.0
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.enums import EnumSchedulerStatus


class ModelRuntimeSchedulerMetrics(BaseModel):
    """Metrics for runtime scheduler operation.

    Tracks tick emission performance, reliability, and operational state
    of the ONEX runtime scheduler. These metrics are essential for
    monitoring scheduler health and identifying performance issues.

    Attributes:
        scheduler_id: Unique identifier for this scheduler instance
        status: Current operational status of the scheduler
        ticks_emitted: Total number of ticks successfully emitted
        ticks_failed: Total number of tick emissions that failed
        last_tick_at: Timestamp of the most recent tick emission
        last_tick_duration_ms: Duration of the most recent tick in milliseconds
        average_tick_duration_ms: Rolling average tick duration in milliseconds
        max_tick_duration_ms: Maximum tick duration observed in milliseconds
        current_sequence_number: Current tick sequence number
        last_persisted_sequence: Last sequence number persisted for restart-safety
        circuit_breaker_open: Whether the circuit breaker is currently open
        consecutive_failures: Number of consecutive tick failures
        started_at: Timestamp when the scheduler started
        total_uptime_seconds: Total scheduler uptime in seconds

    Example:
        >>> from datetime import datetime, timezone
        >>> metrics = ModelRuntimeSchedulerMetrics(
        ...     scheduler_id="prod-scheduler-001",
        ...     ticks_emitted=1000,
        ...     ticks_failed=5,
        ...     average_tick_duration_ms=0.5,
        ... )
        >>> metrics.tick_success_rate()
        0.995...
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    # Scheduler identification
    scheduler_id: str = Field(
        ...,
        description="Unique identifier for this scheduler instance",
        min_length=1,
    )

    # Current status
    status: EnumSchedulerStatus = Field(
        default=EnumSchedulerStatus.STOPPED,
        description="Current operational status of the scheduler",
    )

    # Tick counts
    ticks_emitted: int = Field(
        default=0,
        ge=0,
        description="Total number of ticks successfully emitted",
    )
    ticks_failed: int = Field(
        default=0,
        ge=0,
        description="Total number of tick emissions that failed",
    )

    # Timing metrics (milliseconds)
    last_tick_at: datetime | None = Field(
        default=None,
        description="Timestamp of the most recent tick emission",
    )
    last_tick_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Duration of the most recent tick in milliseconds",
    )
    average_tick_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Rolling average tick duration in milliseconds",
    )
    max_tick_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum tick duration observed in milliseconds",
    )

    # Sequence tracking (for restart-safety)
    current_sequence_number: int = Field(
        default=0,
        ge=0,
        description="Current tick sequence number",
    )
    last_persisted_sequence: int = Field(
        default=0,
        ge=0,
        description="Last sequence number persisted for restart-safety",
    )

    # Circuit breaker state
    circuit_breaker_open: bool = Field(
        default=False,
        description="Whether the circuit breaker is currently open",
    )
    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive tick failures",
    )

    # Uptime tracking
    started_at: datetime | None = Field(
        default=None,
        description="Timestamp when the scheduler started",
    )
    total_uptime_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total scheduler uptime in seconds",
    )

    def tick_success_rate(self) -> float:
        """
        Calculate the tick success rate.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no ticks have been attempted.

        Example:
            >>> metrics = ModelRuntimeSchedulerMetrics(
            ...     scheduler_id="test",
            ...     ticks_emitted=95,
            ...     ticks_failed=5,
            ... )
            >>> metrics.tick_success_rate()
            0.95
        """
        total = self.ticks_emitted + self.ticks_failed
        if total == 0:
            return 1.0
        return self.ticks_emitted / total

    def is_healthy(self) -> bool:
        """
        Check if the scheduler is in a healthy state.

        A scheduler is considered healthy if:
        - It is in RUNNING status
        - The circuit breaker is closed
        - Consecutive failures are below threshold (5)

        Returns:
            True if the scheduler is healthy, False otherwise

        Example:
            >>> metrics = ModelRuntimeSchedulerMetrics(
            ...     scheduler_id="test",
            ...     status=EnumSchedulerStatus.RUNNING,
            ... )
            >>> metrics.is_healthy()
            True
        """
        return (
            self.status == EnumSchedulerStatus.RUNNING
            and not self.circuit_breaker_open
            and self.consecutive_failures < 5
        )

    def unpersisted_sequence_count(self) -> int:
        """
        Calculate the number of sequences not yet persisted.

        This is important for restart-safety to ensure no ticks are lost.

        Returns:
            Number of sequences that have been emitted but not persisted

        Example:
            >>> metrics = ModelRuntimeSchedulerMetrics(
            ...     scheduler_id="test",
            ...     current_sequence_number=100,
            ...     last_persisted_sequence=95,
            ... )
            >>> metrics.unpersisted_sequence_count()
            5
        """
        return self.current_sequence_number - self.last_persisted_sequence


__all__: list[str] = ["ModelRuntimeSchedulerMetrics"]

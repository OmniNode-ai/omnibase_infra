# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatcher Metrics Model.

Per-dispatcher metrics for dispatch engine observability. Tracks execution counts,
error rates, and latency statistics for individual dispatchers.

Design Pattern:
    ModelDispatcherMetrics is a mutable data model that accumulates metrics during
    dispatch engine operation. Unlike most ONEX models, it is NOT frozen because
    metrics need to be updated in real-time during dispatch operations.

    For thread-safety in production, the dispatch engine maintains its own
    synchronization when updating these metrics.

Thread Safety:
    This model is NOT thread-safe on its own. The MessageDispatchEngine provides
    thread-safety guarantees during metrics collection.

Example:
    >>> from omnibase_infra.models.dispatch import ModelDispatcherMetrics
    >>>
    >>> metrics = ModelDispatcherMetrics(dispatcher_id="user-event-dispatcher")
    >>> metrics = metrics.record_execution(duration_ms=45.2, success=True)
    >>> print(f"Success rate: {metrics.success_rate:.1%}")

See Also:
    omnibase_infra.models.dispatch.ModelDispatchMetrics: Aggregate dispatch metrics
    omnibase_core.runtime.MessageDispatchEngine: Uses these for observability
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelDispatcherMetrics(BaseModel):
    """
    Metrics for a single dispatcher in the dispatch engine.

    Tracks execution statistics including counts, error rates, and latency
    for observability and performance monitoring.

    Attributes:
        dispatcher_id: The dispatcher's unique identifier.
        execution_count: Total number of times this dispatcher was executed.
        success_count: Number of successful executions.
        error_count: Number of failed executions.
        total_latency_ms: Cumulative latency across all executions.
        min_latency_ms: Minimum observed latency (-1.0 if no executions).
        max_latency_ms: Maximum observed latency (-1.0 if no executions).
        last_error_message: Most recent error message (empty string if none).
        last_execution_topic: Topic of the most recent execution (empty string if none).

    Example:
        >>> metrics = ModelDispatcherMetrics(dispatcher_id="my-dispatcher")
        >>> metrics = metrics.record_execution(duration_ms=50.0, success=True)
        >>> metrics.avg_latency_ms
        50.0
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
    )

    # ---- Dispatcher Identity ----
    dispatcher_id: str = Field(
        ...,
        description="The dispatcher's unique identifier.",
        min_length=1,
    )

    # ---- Execution Counts ----
    execution_count: int = Field(
        default=0,
        description="Total number of times this dispatcher was executed.",
        ge=0,
    )
    success_count: int = Field(
        default=0,
        description="Number of successful executions.",
        ge=0,
    )
    error_count: int = Field(
        default=0,
        description="Number of failed executions.",
        ge=0,
    )

    # ---- Latency Metrics ----
    total_latency_ms: float = Field(
        default=0.0,
        description="Cumulative latency across all executions in milliseconds.",
        ge=0,
    )
    min_latency_ms: float = Field(
        default=-1.0,
        description=(
            "Minimum observed latency in milliseconds. "
            "Value of -1.0 indicates no executions recorded yet."
        ),
    )
    max_latency_ms: float = Field(
        default=-1.0,
        description=(
            "Maximum observed latency in milliseconds. "
            "Value of -1.0 indicates no executions recorded yet."
        ),
    )

    # ---- Last Execution Info ----
    last_error_message: str = Field(
        default="",
        description=(
            "Most recent error message. Empty string indicates no errors recorded."
        ),
    )
    last_execution_topic: str = Field(
        default="",
        description=(
            "Topic of the most recent execution. "
            "Empty string indicates no executions recorded yet."
        ),
    )

    @property
    def avg_latency_ms(self) -> float:
        """
        Calculate average latency across all executions.

        Returns:
            Average latency in milliseconds, or 0.0 if no executions.

        Example:
            >>> metrics = ModelDispatcherMetrics(
            ...     dispatcher_id="test",
            ...     execution_count=10,
            ...     total_latency_ms=500.0,
            ... )
            >>> metrics.avg_latency_ms
            50.0
        """
        if self.execution_count == 0:
            return 0.0
        return self.total_latency_ms / self.execution_count

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate as a fraction (0.0 to 1.0).

        Returns:
            Success rate as a decimal, or 1.0 if no executions.

        Example:
            >>> metrics = ModelDispatcherMetrics(
            ...     dispatcher_id="test",
            ...     execution_count=100,
            ...     success_count=95,
            ...     error_count=5,
            ... )
            >>> metrics.success_rate
            0.95
        """
        if self.execution_count == 0:
            return 1.0
        return self.success_count / self.execution_count

    @property
    def error_rate(self) -> float:
        """
        Calculate error rate as a fraction (0.0 to 1.0).

        Returns:
            Error rate as a decimal, or 0.0 if no executions.

        Example:
            >>> metrics = ModelDispatcherMetrics(
            ...     dispatcher_id="test",
            ...     execution_count=100,
            ...     error_count=5,
            ... )
            >>> metrics.error_rate
            0.05
        """
        if self.execution_count == 0:
            return 0.0
        return self.error_count / self.execution_count

    def record_execution(
        self,
        duration_ms: float,
        success: bool,
        topic: str | None = None,
        error_message: str | None = None,
    ) -> "ModelDispatcherMetrics":
        """
        Record an execution and return updated metrics.

        Creates a new ModelDispatcherMetrics instance with updated statistics.

        Args:
            duration_ms: Execution duration in milliseconds.
            success: Whether the execution was successful.
            topic: Optional topic the message was from.
            error_message: Optional error message if execution failed.

        Returns:
            New ModelDispatcherMetrics with updated statistics.

        Example:
            >>> metrics = ModelDispatcherMetrics(dispatcher_id="test")
            >>> metrics = metrics.record_execution(
            ...     duration_ms=45.0,
            ...     success=True,
            ...     topic="user.events.v1",
            ... )
            >>> metrics.execution_count
            1
        """
        # Sentinel value -1.0 means "not yet computed"
        new_min = (
            duration_ms
            if self.min_latency_ms < 0
            else min(self.min_latency_ms, duration_ms)
        )
        new_max = (
            duration_ms
            if self.max_latency_ms < 0
            else max(self.max_latency_ms, duration_ms)
        )

        # Use model_copy to prevent field drift when new fields are added
        return self.model_copy(
            update={
                "execution_count": self.execution_count + 1,
                "success_count": self.success_count + (1 if success else 0),
                "error_count": self.error_count + (0 if success else 1),
                "total_latency_ms": self.total_latency_ms + duration_ms,
                "min_latency_ms": new_min,
                "max_latency_ms": new_max,
                # On failure: use error_message (or empty string); on success: preserve previous
                "last_error_message": (error_message or "")
                if not success
                else self.last_error_message,
                "last_execution_topic": topic if topic else self.last_execution_topic,
            }
        )

    def to_dict(self) -> dict[str, object]:
        """
        Convert to dictionary with computed properties included.

        Returns:
            Dictionary with all metrics including computed properties.
            Values are float, int, or str types.
        """
        return {
            "dispatcher_id": self.dispatcher_id,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "last_error_message": self.last_error_message,
            "last_execution_topic": self.last_execution_topic,
        }


__all__ = ["ModelDispatcherMetrics"]

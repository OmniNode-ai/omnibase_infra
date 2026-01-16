# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dead Letter Queue metrics model for operational visibility.

Provides a Pydantic model for tracking operational metrics of the dead letter
queue, enabling monitoring of:
- DLQ publish success/failure rates
- Per-topic breakdown of DLQ messages
- Per-error-type breakdown for debugging
- Latency statistics for DLQ operations

Design Pattern:
    Like ModelDispatchMetrics, this model uses the copy-on-write pattern.
    Update methods return NEW instances rather than mutating in place:

    ```python
    # Copy-on-write: returns new instance, does not mutate
    metrics = metrics.record_dlq_publish(
        original_topic="orders.created",
        error_type="ValidationError",
        success=True,
        duration_ms=15.5,
    )
    ```

Thread Safety:
    Individual ModelDlqMetrics instances are safe to share across threads
    since update operations return new instances. The EventBusKafka uses
    a lock to ensure atomic read-modify-write cycles when updating the
    shared metrics reference.

Example:
    >>> from omnibase_infra.event_bus.models import ModelDlqMetrics
    >>>
    >>> metrics = ModelDlqMetrics()
    >>> metrics = metrics.record_dlq_publish(
    ...     original_topic="orders.created",
    ...     error_type="ValidationError",
    ...     success=True,
    ...     duration_ms=15.5,
    ... )
    >>> print(f"DLQ success rate: {metrics.success_rate:.1%}")
    DLQ success rate: 100.0%

See Also:
    ModelDlqEvent: Individual DLQ event for callbacks
    ModelDispatchMetrics: Similar pattern for dispatch metrics
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelDlqMetrics(BaseModel):
    """Aggregate metrics for dead letter queue operations.

    Provides comprehensive observability for DLQ health including:
    - Total publish counts and success/failure rates
    - Per-topic metrics breakdown
    - Per-error-type metrics breakdown
    - Latency statistics for DLQ publish operations
    - Last operation timestamps for staleness detection

    Attributes:
        total_publishes: Total number of DLQ publish attempts
        successful_publishes: Number of successful DLQ publishes
        failed_publishes: Number of failed DLQ publishes
        topic_counts: Per-original-topic DLQ message counts
        error_type_counts: Per-error-type DLQ message counts
        total_latency_ms: Cumulative DLQ publish latency in milliseconds
        min_latency_ms: Minimum observed DLQ publish latency
        max_latency_ms: Maximum observed DLQ publish latency
        last_publish_at: Timestamp of the last DLQ publish attempt
        last_failure_at: Timestamp of the last failed DLQ publish (if any)

    Example:
        >>> metrics = ModelDlqMetrics()
        >>> print(f"Total DLQ publishes: {metrics.total_publishes}")
        >>> print(f"Success rate: {metrics.success_rate:.1%}")
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
    )

    # ---- Publish Counts ----
    total_publishes: int = Field(
        default=0,
        description="Total number of DLQ publish attempts",
        ge=0,
    )
    successful_publishes: int = Field(
        default=0,
        description="Number of successful DLQ publishes",
        ge=0,
    )
    failed_publishes: int = Field(
        default=0,
        description="Number of failed DLQ publishes",
        ge=0,
    )

    # ---- Per-Topic Metrics ----
    topic_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-original-topic DLQ message counts. Keys are topic names, "
            "values are the number of messages from that topic sent to DLQ."
        ),
    )

    # ---- Per-Error-Type Metrics ----
    error_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-error-type DLQ message counts. Keys are error type names "
            "(e.g., 'ValidationError'), values are occurrence counts."
        ),
    )

    # ---- Latency Statistics ----
    total_latency_ms: float = Field(
        default=0.0,
        description="Cumulative DLQ publish latency in milliseconds",
        ge=0,
    )
    min_latency_ms: float | None = Field(
        default=None,
        description="Minimum observed DLQ publish latency in milliseconds",
    )
    max_latency_ms: float | None = Field(
        default=None,
        description="Maximum observed DLQ publish latency in milliseconds",
    )

    # ---- Timestamps ----
    last_publish_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last DLQ publish attempt",
    )
    last_failure_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last failed DLQ publish (if any)",
    )

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average DLQ publish latency.

        Returns:
            Average latency in milliseconds, or 0.0 if no publishes.
        """
        if self.total_publishes == 0:
            return 0.0
        return self.total_latency_ms / self.total_publishes

    @property
    def success_rate(self) -> float:
        """Calculate DLQ publish success rate.

        Returns:
            Success rate as a decimal (0.0 to 1.0), or 1.0 if no publishes.
        """
        if self.total_publishes == 0:
            return 1.0
        return self.successful_publishes / self.total_publishes

    @property
    def failure_rate(self) -> float:
        """Calculate DLQ publish failure rate.

        Returns:
            Failure rate as a decimal (0.0 to 1.0), or 0.0 if no publishes.
        """
        if self.total_publishes == 0:
            return 0.0
        return self.failed_publishes / self.total_publishes

    def record_dlq_publish(
        self,
        original_topic: str,
        error_type: str,
        success: bool,
        duration_ms: float,
    ) -> ModelDlqMetrics:
        """Record a DLQ publish operation and return updated metrics.

        Creates a new ModelDlqMetrics instance with updated statistics.

        Args:
            original_topic: The topic where the original message was consumed
            error_type: The type name of the exception that caused DLQ
            success: Whether the DLQ publish succeeded
            duration_ms: Duration of the DLQ publish operation in milliseconds

        Returns:
            New ModelDlqMetrics with updated statistics
        """
        now = datetime.now(UTC)

        # Update latency statistics
        new_min = (
            duration_ms
            if self.min_latency_ms is None
            else min(self.min_latency_ms, duration_ms)
        )
        new_max = (
            duration_ms
            if self.max_latency_ms is None
            else max(self.max_latency_ms, duration_ms)
        )

        # Update topic counts
        new_topic_counts = dict(self.topic_counts)
        new_topic_counts[original_topic] = new_topic_counts.get(original_topic, 0) + 1

        # Update error type counts
        new_error_type_counts = dict(self.error_type_counts)
        new_error_type_counts[error_type] = new_error_type_counts.get(error_type, 0) + 1

        # Determine timestamps
        new_last_failure_at = self.last_failure_at
        if not success:
            new_last_failure_at = now

        return self.model_copy(
            update={
                "total_publishes": self.total_publishes + 1,
                "successful_publishes": self.successful_publishes
                + (1 if success else 0),
                "failed_publishes": self.failed_publishes + (0 if success else 1),
                "topic_counts": new_topic_counts,
                "error_type_counts": new_error_type_counts,
                "total_latency_ms": self.total_latency_ms + duration_ms,
                "min_latency_ms": new_min,
                "max_latency_ms": new_max,
                "last_publish_at": now,
                "last_failure_at": new_last_failure_at,
            }
        )

    def get_topic_count(self, topic: str) -> int:
        """Get DLQ message count for a specific original topic.

        Args:
            topic: The original topic name

        Returns:
            Number of messages from this topic sent to DLQ
        """
        return self.topic_counts.get(topic, 0)

    def get_error_type_count(self, error_type: str) -> int:
        """Get DLQ message count for a specific error type.

        Args:
            error_type: The error type name (e.g., 'ValidationError')

        Returns:
            Number of messages sent to DLQ due to this error type
        """
        return self.error_type_counts.get(error_type, 0)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary with computed properties included.

        Returns:
            Dictionary with all metrics including computed properties
        """
        return {
            "total_publishes": self.total_publishes,
            "successful_publishes": self.successful_publishes,
            "failed_publishes": self.failed_publishes,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "topic_counts": self.topic_counts,
            "error_type_counts": self.error_type_counts,
            "last_publish_at": (
                self.last_publish_at.isoformat() if self.last_publish_at else None
            ),
            "last_failure_at": (
                self.last_failure_at.isoformat() if self.last_failure_at else None
            ),
        }

    @classmethod
    def create_empty(cls) -> ModelDlqMetrics:
        """Create a new empty metrics instance.

        Returns:
            New ModelDlqMetrics with all counters at zero
        """
        return cls()


__all__: list[str] = ["ModelDlqMetrics"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Transition Notification Publisher Metrics Model.

This module provides the Pydantic model for tracking transition notification
publisher performance and operational metrics. Used for monitoring notification
delivery reliability, timing performance, and publisher health.

Metrics Categories:
    - Notification Counts: Total notifications published (single and batch)
    - Timing Metrics: Duration tracking for publish operations
    - Error Tracking: Failed notifications and error rates
    - Circuit Breaker: Failure tolerance state

Thread Safety:
    This model is immutable (frozen=True) and safe to share across threads.
    Create new instances for updated metrics using model_copy(update={...}).

Example:
    >>> from datetime import datetime, UTC
    >>> metrics = ModelTransitionNotificationPublisherMetrics(
    ...     publisher_id="publisher-001",
    ...     topic="onex.fsm.state.transitions.v1",
    ...     notifications_published=100,
    ...     last_publish_at=datetime.now(UTC),
    ... )
    >>> print(metrics.publish_success_rate())
    1.0
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ModelTransitionNotificationPublisherMetrics(BaseModel):
    """Metrics for transition notification publisher operation.

    Tracks notification publishing performance, reliability, and operational
    state of the ONEX transition notification publisher. These metrics are
    essential for monitoring publisher health and identifying performance issues.

    Attributes:
        publisher_id: Unique identifier for this publisher instance
        topic: Target topic for notifications
        notifications_published: Total notifications successfully published
        notifications_failed: Total notifications that failed to publish
        batch_operations: Total batch publish operations executed
        batch_notifications_total: Total notifications published via batch
        last_publish_at: Timestamp of the most recent publish operation
        last_publish_duration_ms: Duration of the most recent publish in ms
        average_publish_duration_ms: Rolling average publish duration in ms
        max_publish_duration_ms: Maximum publish duration observed in ms
        circuit_breaker_open: Whether the circuit breaker is currently open
        consecutive_failures: Number of consecutive publish failures
        started_at: Timestamp when the publisher started

    Note:
        Notification Count Relationships:
        - ``notifications_published`` counts ALL successful publishes (individual + batch)
        - ``batch_notifications_total`` is a SUBSET of ``notifications_published``,
          counting only those published via ``publish_batch()``
        - Individual publishes = ``notifications_published - batch_notifications_total``

    Example:
        >>> from datetime import datetime, UTC
        >>> metrics = ModelTransitionNotificationPublisherMetrics(
        ...     publisher_id="prod-publisher-001",
        ...     topic="onex.fsm.state.transitions.v1",
        ...     notifications_published=1000,
        ...     notifications_failed=5,
        ...     average_publish_duration_ms=1.5,
        ... )
        >>> metrics.publish_success_rate()
        0.995...
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    # Health check threshold - matches default circuit breaker failure threshold.
    # When consecutive_failures reaches this value, the circuit breaker opens,
    # so is_healthy() returns False for both the threshold check and circuit state.
    DEFAULT_HEALTH_FAILURE_THRESHOLD: ClassVar[int] = 5

    # Publisher identification
    publisher_id: str = Field(
        ...,
        description="Unique identifier for this publisher instance",
        min_length=1,
    )

    topic: str = Field(
        ...,
        description="Target topic for transition notifications",
        min_length=1,
    )

    # Notification counts (includes both individual and batch publishes)
    notifications_published: int = Field(
        default=0,
        ge=0,
        description=(
            "Total notifications successfully published (ALL publishes). "
            "Includes both individual publish() and batch publish_batch() calls. "
            "Individual publishes = notifications_published - batch_notifications_total."
        ),
    )
    notifications_failed: int = Field(
        default=0,
        ge=0,
        description="Total number of notifications that failed to publish",
    )

    # Batch operation counts
    batch_operations: int = Field(
        default=0,
        ge=0,
        description="Total number of batch publish operations executed",
    )
    batch_notifications_total: int = Field(
        default=0,
        ge=0,
        description=(
            "Notifications published via publish_batch() (SUBSET of notifications_published). "
            "This count is already included in notifications_published, not additional. "
            "Example: 100 published, 80 batch_total means 20 individual + 80 batch = 100 total."
        ),
    )

    # Timing metrics (milliseconds)
    last_publish_at: datetime | None = Field(
        default=None,
        description="Timestamp of the most recent publish operation",
    )
    last_publish_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Duration of the most recent publish in milliseconds",
    )
    average_publish_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Rolling average publish duration in milliseconds",
    )
    max_publish_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum publish duration observed in milliseconds",
    )

    # Circuit breaker state
    circuit_breaker_open: bool = Field(
        default=False,
        description="Whether the circuit breaker is currently open",
    )
    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive publish failures",
    )

    # Lifecycle tracking
    started_at: datetime | None = Field(
        default=None,
        description="Timestamp when the publisher started",
    )

    def publish_success_rate(self) -> float:
        """
        Calculate the publish success rate.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no notifications have been attempted.

        Example:
            >>> metrics = ModelTransitionNotificationPublisherMetrics(
            ...     publisher_id="test",
            ...     topic="test.topic",
            ...     notifications_published=95,
            ...     notifications_failed=5,
            ... )
            >>> metrics.publish_success_rate()
            0.95
        """
        total = self.notifications_published + self.notifications_failed
        if total == 0:
            return 1.0
        return self.notifications_published / total

    def is_healthy(self) -> bool:
        """
        Check if the publisher is in a healthy state.

        A publisher is considered healthy if:
        - The circuit breaker is closed
        - Consecutive failures are below the health threshold

        Note:
            The health threshold (DEFAULT_HEALTH_FAILURE_THRESHOLD = 5) matches
            the default circuit breaker failure threshold. This alignment means
            that when consecutive failures reach this count, the circuit breaker
            opens and is_healthy() returns False for both conditions. This
            provides early warning before the circuit breaker triggers, since
            the threshold check fails at the same point the breaker would open.

        Returns:
            True if the publisher is healthy, False otherwise

        Example:
            >>> metrics = ModelTransitionNotificationPublisherMetrics(
            ...     publisher_id="test",
            ...     topic="test.topic",
            ... )
            >>> metrics.is_healthy()
            True
        """
        return (
            not self.circuit_breaker_open
            and self.consecutive_failures < self.DEFAULT_HEALTH_FAILURE_THRESHOLD
        )

    def total_notifications(self) -> int:
        """
        Get total notification publish count.

        Note: notifications_published includes all notifications (both individual
        and batch). batch_notifications_total is a subset indicating how many
        were published via batch operations.

        Returns:
            Total number of notifications published

        Example:
            >>> metrics = ModelTransitionNotificationPublisherMetrics(
            ...     publisher_id="test",
            ...     topic="test.topic",
            ...     notifications_published=100,  # Total published
            ...     batch_notifications_total=80,  # Subset via batch
            ... )
            >>> metrics.total_notifications()
            100
        """
        return self.notifications_published


__all__: list[str] = ["ModelTransitionNotificationPublisherMetrics"]

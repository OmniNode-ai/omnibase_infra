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
        batch_notifications_attempted: Total notifications attempted via batch
        batch_notifications_total: Total notifications successfully published via batch
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
        - ``batch_notifications_attempted`` counts ALL notifications passed to ``publish_batch()``,
          regardless of success or failure (sum of all batch sizes)
        - ``batch_notifications_total`` is a SUBSET of ``notifications_published``,
          counting only those SUCCESSFULLY published via ``publish_batch()``
        - Individual publishes = ``notifications_published - batch_notifications_total``
        - Batch failure rate = ``1 - (batch_notifications_total / batch_notifications_attempted)``

    Derived Metrics Calculations:
        The model provides convenience methods for calculating derived metrics from
        the raw counters. These formulas are useful for monitoring and alerting:

        **Batch Failure Rate** (via ``batch_failure_rate()`` method):
            Formula: ``1 - (batch_notifications_total / batch_notifications_attempted)``
            - Only valid when ``batch_notifications_attempted > 0``
            - Returns 0.0 when no batch operations have been attempted
            - A rate of 0.0 means all batch notifications succeeded
            - A rate of 1.0 means all batch notifications failed

        **Individual Publishes Count** (via ``individual_publish_count()`` method):
            Formula: ``notifications_published - batch_notifications_total``
            - Represents the count of single (non-batch) publish operations
            - Always non-negative since batch_notifications_total is a subset

        **Overall Failure Rate** (inverse of ``publish_success_rate()``):
            Formula: ``notifications_failed / (notifications_published + notifications_failed)``
            - Only valid when total attempts > 0
            - Returns 0.0 when no notifications have been attempted
            - Covers both individual and batch publish failures

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

    # Health check threshold - intentionally matches default circuit breaker threshold.
    #
    # Design Decision: This value is hardcoded rather than configurable because:
    # 1. The health threshold MUST match the circuit breaker threshold for consistent
    #    behavior - when consecutive_failures reaches this value, the circuit opens
    # 2. Making this configurable independently would break the correlation between
    #    is_healthy() returning False and the circuit breaker opening
    # 3. If you need a different threshold, configure both circuit_breaker_threshold
    #    in TransitionNotificationPublisher and accept that health checks will match
    #
    # The is_healthy() method checks both circuit_breaker_open AND consecutive_failures
    # to provide early warning before the circuit actually opens (at N-1 failures).
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
    batch_notifications_attempted: int = Field(
        default=0,
        ge=0,
        description=(
            "Total notifications attempted via publish_batch() calls (includes failures). "
            "This is the sum of all batch sizes passed to publish_batch(), regardless of outcome. "
            "Formula: batch_failure_rate = 1 - (batch_notifications_total / batch_notifications_attempted)."
        ),
    )
    batch_notifications_total: int = Field(
        default=0,
        ge=0,
        description=(
            "Successful batch publishes only (SUBSET of notifications_published). "
            "Counts notifications that succeeded via publish_batch(), not total attempts. "
            "Already included in notifications_published (not additional). "
            "Formula: individual_publishes = notifications_published - batch_notifications_total."
        ),
    )
    batch_failures_truncated: int = Field(
        default=0,
        ge=0,
        description=(
            "Count of times failure tracking was truncated due to exceeding max_tracked_failures. "
            "When batch operations have more failures than max_tracked_failures (default 100), "
            "only the first max_tracked_failures are stored in memory. This counter tracks "
            "how many times truncation occurred, indicating potential memory pressure events."
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

    def batch_failure_rate(self) -> float:
        """
        Calculate the batch publish failure rate.

        This metric indicates the proportion of notifications that failed when
        publishing via batch operations. A high batch failure rate may indicate
        issues with the message broker, serialization problems, or network
        instability during batch operations.

        Formula: ``1 - (batch_notifications_total / batch_notifications_attempted)``

        Returns:
            Failure rate as a float between 0.0 and 1.0.
            Returns 0.0 if no batch operations have been attempted.

        Note:
            - A rate of 0.0 means all batch notifications succeeded
            - A rate of 1.0 means all batch notifications failed
            - This only covers batch operations; individual publish failures
              are tracked separately via ``publish_success_rate()``

        Example:
            >>> metrics = ModelTransitionNotificationPublisherMetrics(
            ...     publisher_id="test",
            ...     topic="test.topic",
            ...     batch_notifications_attempted=100,
            ...     batch_notifications_total=95,
            ... )
            >>> metrics.batch_failure_rate()
            0.05
            >>> # 5% of batch notifications failed (5 out of 100)
        """
        if self.batch_notifications_attempted == 0:
            return 0.0
        return 1.0 - (
            self.batch_notifications_total / self.batch_notifications_attempted
        )

    def individual_publish_count(self) -> int:
        """
        Calculate the count of individual (non-batch) publish operations.

        This metric represents the number of notifications that were published
        via single ``publish()`` calls rather than batch ``publish_batch()`` calls.
        Useful for understanding the distribution of publish patterns and
        optimizing batch usage.

        Formula: ``notifications_published - batch_notifications_total``

        Returns:
            Count of individual publishes as a non-negative integer.

        Note:
            This value is always non-negative because ``batch_notifications_total``
            is a subset of ``notifications_published`` (successful batch publishes
            are counted in both fields).

        Example:
            >>> metrics = ModelTransitionNotificationPublisherMetrics(
            ...     publisher_id="test",
            ...     topic="test.topic",
            ...     notifications_published=150,
            ...     batch_notifications_total=100,
            ... )
            >>> metrics.individual_publish_count()
            50
            >>> # 50 notifications were published individually, 100 via batch
        """
        return self.notifications_published - self.batch_notifications_total


__all__: list[str] = ["ModelTransitionNotificationPublisherMetrics"]

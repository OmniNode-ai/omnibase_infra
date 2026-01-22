# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Transition Notification Outbox Metrics Model.

This module provides a strongly-typed Pydantic model for outbox metrics,
replacing the untyped dict return from get_metrics().
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ModelTransitionNotificationOutboxMetrics(BaseModel):
    """Metrics for transition notification outbox operation.

    This model provides type-safe access to outbox metrics for observability
    and monitoring purposes.

    Attributes:
        table_name: The outbox table name.
        is_running: Whether the background processor is currently running.
        notifications_stored: Total count of notifications stored in outbox.
        notifications_processed: Total count of notifications successfully processed.
        notifications_failed: Total count of notifications that failed processing.
        notifications_sent_to_dlq: Total count of notifications sent to DLQ after
            exceeding max retry attempts.
        dlq_publish_failures: Count of failed DLQ publish attempts.
        batch_size: Configured batch size for processing.
        poll_interval_seconds: Configured poll interval in seconds.
        max_retries: Configured maximum retry attempts before DLQ (None if DLQ disabled).
        dlq_topic: Configured DLQ topic name (None if DLQ disabled).

    Class Variables:
        DEFAULT_DLQ_ALERT_THRESHOLD: Recommended threshold for alerting on DLQ
            unavailability. Non-zero dlq_publish_failures indicate the DLQ is
            having issues; reaching this threshold suggests operator intervention
            is needed.

    Observability Helpers:
        dlq_needs_attention(): Returns True when DLQ publish failures have reached
            the alert threshold, indicating operator intervention may be needed.
        pending_dlq_ratio(): Returns the ratio of notifications stuck in DLQ retry
            state, helping operators understand what fraction of failures are
            DLQ-related versus normal processing failures.
    """

    DEFAULT_DLQ_ALERT_THRESHOLD: ClassVar[int] = 3
    """Recommended threshold for alerting on DLQ unavailability.

    Non-zero dlq_publish_failures indicate the DLQ is having issues. When this
    threshold is reached, it suggests the DLQ has been unavailable for multiple
    consecutive attempts and operator intervention is needed to investigate:
    - Network connectivity to the DLQ broker
    - DLQ topic existence and permissions
    - Broker availability and health
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    table_name: str = Field(..., description="The outbox table name")
    is_running: bool = Field(default=False, description="Whether processor is running")
    notifications_stored: int = Field(
        default=0, ge=0, description="Total notifications stored"
    )
    notifications_processed: int = Field(
        default=0, ge=0, description="Total notifications successfully processed"
    )
    notifications_failed: int = Field(
        default=0, ge=0, description="Total notifications that failed processing"
    )
    notifications_sent_to_dlq: int = Field(
        default=0, ge=0, description="Total notifications sent to DLQ"
    )
    dlq_publish_failures: int = Field(
        default=0,
        ge=0,
        description=(
            "Count of failed DLQ publish attempts. Non-zero values indicate DLQ "
            "availability issues. Monitor this metric to detect when the DLQ is "
            "unavailable, which can cause infinite retry loops for notifications "
            "that have exceeded max_retries."
        ),
    )
    batch_size: int = Field(default=100, ge=1, description="Configured batch size")
    poll_interval_seconds: float = Field(
        default=1.0, gt=0, description="Configured poll interval"
    )
    max_retries: int | None = Field(
        default=None, ge=1, description="Max retries before DLQ (None if DLQ disabled)"
    )
    dlq_topic: str | None = Field(
        default=None, description="DLQ topic name (None if DLQ disabled)"
    )

    def dlq_needs_attention(self) -> bool:
        """Check if DLQ publish failures have reached the alert threshold.

        This method indicates when the DLQ is experiencing availability issues
        that may require operator intervention. When True, it suggests:

        - The DLQ has been unavailable for multiple consecutive attempts
        - Notifications that exceeded max_retries are stuck in retry loops
        - Operator should investigate DLQ broker connectivity and health

        Returns:
            False if DLQ is disabled (max_retries is None).
            True if dlq_publish_failures >= DEFAULT_DLQ_ALERT_THRESHOLD.
            False otherwise (DLQ is enabled but failures below threshold).
        """
        if self.max_retries is None:
            return False
        return self.dlq_publish_failures >= self.DEFAULT_DLQ_ALERT_THRESHOLD

    def pending_dlq_ratio(self) -> float:
        """Calculate the ratio of notifications stuck in DLQ retry state.

        This metric helps operators understand what fraction of failed
        notifications are DLQ-related versus normal processing failures.
        A high ratio indicates DLQ availability is the bottleneck.

        Returns:
            0.0 if DLQ is disabled (max_retries is None).
            0.0 if no notifications have failed (notifications_failed == 0).
            Ratio of dlq_publish_failures to notifications_failed otherwise.
            Formula: dlq_publish_failures / max(1, notifications_failed)
        """
        if self.max_retries is None:
            return 0.0
        if self.notifications_failed == 0:
            return 0.0
        return self.dlq_publish_failures / max(1, self.notifications_failed)


__all__: list[str] = ["ModelTransitionNotificationOutboxMetrics"]

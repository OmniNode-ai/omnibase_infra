# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dead Letter Queue event model for callback payloads.

Provides a strongly-typed model for DLQ events that captures all relevant
context for alerting, monitoring, and debugging failed message processing.

This model is passed to DLQ callback hooks when messages are published to
the dead letter queue, enabling custom alerting integration (PagerDuty,
Slack, email, etc.) without coupling the EventBusKafka to specific
alerting implementations.

Example:
    ```python
    from omnibase_infra.event_bus.models import ModelDlqEvent

    async def alert_on_dlq(event: ModelDlqEvent) -> None:
        '''Custom alerting callback for DLQ events.'''
        if event.success:
            logger.warning(
                "Message sent to DLQ",
                extra=event.to_log_context(),
            )
        else:
            # DLQ publish itself failed - critical alert
            pagerduty.trigger(
                summary=f"DLQ publish failed: {event.dlq_error_type}",
                severity="critical",
                details=event.to_log_context(),
            )
    ```

See Also:
    ModelDlqMetrics: Aggregate metrics for DLQ operations
    EventBusKafka._publish_to_dlq: DLQ publishing implementation
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDlqEvent(BaseModel):
    """Event model for dead letter queue publish operations.

    Captures comprehensive context for a DLQ publish event, including:
    - Original message context (topic, offset, partition)
    - Failure information (error type, error message)
    - DLQ publish result (success/failure, DLQ error if failed)
    - Tracing information (correlation_id, timestamps)

    This model follows the ONEX immutable model pattern with frozen=True
    to ensure thread-safe sharing across callback invocations.

    Attributes:
        original_topic: The Kafka topic where the message was originally consumed
        dlq_topic: The dead letter queue topic where the message was published
        correlation_id: Correlation ID for distributed tracing
        error_type: The type name of the exception that caused the original failure
        error_message: The error message from the original failure
        retry_count: Number of retry attempts before DLQ
        message_offset: Original message offset (if available)
        message_partition: Original message partition (if available)
        success: Whether the DLQ publish succeeded
        dlq_error_type: Type of error if DLQ publish failed (None if success)
        dlq_error_message: Error message if DLQ publish failed (None if success)
        timestamp: When the DLQ event occurred
        environment: Environment identifier (e.g., "prod", "staging")
        consumer_group: Consumer group that processed the message

    Example:
        >>> event = ModelDlqEvent(
        ...     original_topic="orders.created",
        ...     dlq_topic="dlq.orders",
        ...     correlation_id=uuid4(),
        ...     error_type="ValidationError",
        ...     error_message="Invalid order format",
        ...     success=True,
        ... )
        >>> print(event.to_log_context())
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Original message context
    original_topic: str = Field(
        description="The Kafka topic where the message was originally consumed",
    )
    dlq_topic: str = Field(
        description="The dead letter queue topic where the message was published",
    )
    correlation_id: UUID = Field(
        description="Correlation ID for distributed tracing",
    )

    # Failure information
    error_type: str = Field(
        description="The type name of the exception that caused the original failure",
    )
    error_message: str = Field(
        description="The error message from the original failure (sanitized)",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts before DLQ",
        ge=0,
    )

    # Original message metadata (optional, may not be available)
    message_offset: str | None = Field(
        default=None,
        description="Original message offset (if available)",
    )
    message_partition: int | None = Field(
        default=None,
        description="Original message partition (if available)",
    )

    # DLQ publish result
    success: bool = Field(
        description="Whether the DLQ publish succeeded",
    )
    dlq_error_type: str | None = Field(
        default=None,
        description="Type of error if DLQ publish failed (None if success)",
    )
    dlq_error_message: str | None = Field(
        default=None,
        description="Error message if DLQ publish failed (None if success)",
    )

    # Tracing and environment
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the DLQ event occurred",
    )
    environment: str = Field(
        default="unknown",
        description="Environment identifier (e.g., 'prod', 'staging')",
    )
    consumer_group: str = Field(
        default="unknown",
        description="Consumer group that processed the message",
    )

    def to_log_context(self) -> dict[str, object]:
        """Convert to a dictionary suitable for structured logging.

        Returns a dictionary with all fields formatted for logging,
        with correlation_id converted to string and timestamp to ISO format.

        Returns:
            Dictionary with all event fields for structured logging
        """
        return {
            "original_topic": self.original_topic,
            "dlq_topic": self.dlq_topic,
            "correlation_id": str(self.correlation_id),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "message_offset": self.message_offset,
            "message_partition": self.message_partition,
            "success": self.success,
            "dlq_error_type": self.dlq_error_type,
            "dlq_error_message": self.dlq_error_message,
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment,
            "consumer_group": self.consumer_group,
        }

    @property
    def is_critical(self) -> bool:
        """Determine if this event represents a critical failure.

        A critical failure is when the DLQ publish itself failed,
        meaning the failed message was lost and cannot be recovered
        from the DLQ for later analysis or retry.

        Returns:
            True if DLQ publish failed (message potentially lost)
        """
        return not self.success

    @property
    def metric_labels(self) -> dict[str, str]:
        """Return labels suitable for metrics emission.

        Provides a consistent set of labels for metrics counters,
        following Prometheus/OpenTelemetry labeling conventions.

        Returns:
            Dictionary of label names to values for metrics
        """
        return {
            "original_topic": self.original_topic,
            "error_type": self.error_type,
            "environment": self.environment,
            "success": str(self.success).lower(),
        }


__all__: list[str] = ["ModelDlqEvent"]

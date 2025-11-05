"""Dead Letter Queue Entry Model.

Strongly-typed model for circuit breaker dead letter queue entries.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from datetime import datetime
from uuid import UUID

from omnibase_core.models.core.model_onex_event import ModelOnexEvent
from pydantic import BaseModel, Field


class ModelDeadLetterQueueEntry(BaseModel):
    """Model for circuit breaker dead letter queue entries."""

    # Entry identification
    entry_id: UUID = Field(
        description="Unique identifier for this dead letter queue entry",
    )

    # Original event
    original_event: ModelOnexEvent = Field(
        description="The original event that failed processing",
    )

    # Failure information
    failure_timestamp: datetime = Field(
        description="When the event failed and was queued",
    )

    failure_reason: str = Field(
        max_length=500,
        description="Reason why the event failed",
    )

    error_type: str | None = Field(
        default=None,
        max_length=100,
        description="Type/class of error that occurred",
    )

    error_message: str | None = Field(
        default=None,
        max_length=1000,
        description="Detailed error message",
    )

    # Retry information
    retry_count: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of times processing has been retried",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries allowed",
    )

    next_retry_at: datetime | None = Field(
        default=None,
        description="When the next retry should be attempted",
    )

    last_retry_at: datetime | None = Field(
        default=None,
        description="When the last retry was attempted",
    )

    # Circuit breaker context
    circuit_breaker_state_when_failed: str = Field(
        pattern="^(CLOSED|HALF_OPEN|OPEN)$",
        description="Circuit breaker state when the event failed",
    )

    failure_count_when_failed: int = Field(
        ge=0,
        description="Circuit breaker failure count when this event failed",
    )

    # Processing context
    original_publisher_function: str | None = Field(
        default=None,
        max_length=200,
        description="Name of the publisher function that originally failed",
    )

    processing_timeout_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Timeout that was applied when processing failed (milliseconds)",
    )

    # Queue management
    queue_position: int | None = Field(
        default=None,
        ge=0,
        description="Position in the dead letter queue (for ordering)",
    )

    expires_at: datetime | None = Field(
        default=None,
        description="When this entry expires and should be removed from queue",
    )

    # Resolution tracking
    resolved: bool = Field(
        default=False,
        description="Whether this entry has been successfully processed",
    )

    resolved_at: datetime | None = Field(
        default=None,
        description="When this entry was successfully processed",
    )

    resolved_by: str | None = Field(
        default=None,
        max_length=100,
        description="How this entry was resolved (retry_success, manual_intervention, etc.)",
    )

    # Metadata
    environment: str | None = Field(
        default=None,
        max_length=50,
        description="Environment where the failure occurred",
    )

    service_version: str | None = Field(
        default=None,
        max_length=50,
        description="Version of the service when failure occurred",
    )

    additional_context: str | None = Field(
        default=None,
        max_length=1000,
        description="Additional context about the failure",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

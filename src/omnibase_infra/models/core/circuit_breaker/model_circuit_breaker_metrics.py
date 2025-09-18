"""Circuit Breaker Metrics Model.

Shared model for circuit breaker metrics and performance data.
Used across circuit breaker nodes and observability systems.
"""

from datetime import datetime

from omnibase_core.models.model_base import ModelBase
from pydantic import Field, field_validator, computed_field


class ModelCircuitBreakerMetrics(ModelBase):
    """Model for circuit breaker metrics tracking."""

    total_events: int = Field(
        default=0,
        ge=0,
        description="Total number of events processed",
    )

    successful_events: int = Field(
        default=0,
        ge=0,
        description="Number of successfully processed events",
    )

    failed_events: int = Field(
        default=0,
        ge=0,
        description="Number of failed events",
    )

    queued_events: int = Field(
        default=0,
        ge=0,
        description="Number of events currently queued",
    )

    dropped_events: int = Field(
        default=0,
        ge=0,
        description="Number of events dropped due to capacity limits",
    )

    dead_letter_events: int = Field(
        default=0,
        ge=0,
        description="Number of events in dead letter queue",
    )

    circuit_opens: int = Field(
        default=0,
        ge=0,
        description="Number of times circuit has opened",
    )

    circuit_closes: int = Field(
        default=0,
        ge=0,
        description="Number of times circuit has closed",
    )

    last_failure: datetime | None = Field(
        default=None,
        description="Timestamp of last failure",
    )

    last_success: datetime | None = Field(
        default=None,
        description="Timestamp of last success",
    )

    @computed_field
    @property
    def success_rate_percent(self) -> float:
        """Calculate success rate percentage with division by zero prevention."""
        if self.total_events == 0:
            return 0.0
        return (self.successful_events / self.total_events) * 100.0

    @field_validator('successful_events')
    @classmethod
    def validate_successful_events_not_exceed_total(cls, v: int, info) -> int:
        """Ensure successful events don't exceed total events."""
        if hasattr(info.data, 'total_events') and info.data['total_events'] > 0:
            if v > info.data['total_events']:
                raise ValueError("Successful events cannot exceed total events")
        return v

    average_response_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average response time in milliseconds",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

"""Circuit Breaker Metrics Model.

Shared model for circuit breaker metrics and performance data.
Used across circuit breaker nodes and observability systems.
"""

from datetime import datetime

from pydantic import BaseModel, Field, computed_field, field_validator


class ModelCircuitBreakerMetrics(BaseModel):
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

    @computed_field
    @property
    def failure_rate_percent(self) -> float:
        """Calculate failure rate percentage as complement of success rate."""
        return 100.0 - self.success_rate_percent

    @computed_field
    @property
    def total_error_events(self) -> int:
        """Calculate total error events (failed + dropped + dead letter)."""
        return self.failed_events + self.dropped_events + self.dead_letter_events

    @computed_field
    @property
    def circuit_stability_ratio(self) -> float:
        """Calculate circuit stability ratio (closes vs opens)."""
        if self.circuit_opens == 0:
            return float("inf") if self.circuit_closes > 0 else 1.0
        return self.circuit_closes / self.circuit_opens

    @computed_field
    @property
    def queue_utilization_percent(self) -> float:
        """Calculate queue utilization as percentage of total events."""
        if self.total_events == 0:
            return 0.0
        return (self.queued_events / self.total_events) * 100.0

    @field_validator("successful_events")
    @classmethod
    def validate_successful_events_not_exceed_total(cls, v: int, info) -> int:
        """Ensure successful events don't exceed total events."""
        if hasattr(info.data, "total_events") and info.data["total_events"] > 0:
            if v > info.data["total_events"]:
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

"""Circuit Breaker Operation Results Models.

Strongly-typed models for different circuit breaker operation results.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from omnibase_core.enums.intelligence.enum_circuit_breaker_state import EnumCircuitBreakerState


class ModelPublishEventResult(BaseModel):
    """Result for publish event operations."""

    event_published: bool = Field(
        description="Whether the event was successfully published"
    )

    publisher_function: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Name of the publisher function used"
    )

    publish_latency_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time taken to publish event in milliseconds"
    )

    circuit_action_taken: str = Field(
        pattern="^(published|queued|dropped|rejected)$",
        description="Action taken by circuit breaker"
    )

    queue_length_after: Optional[int] = Field(
        default=None,
        ge=0,
        description="Length of event queue after operation"
    )

    dead_letter_queued: bool = Field(
        default=False,
        description="Whether event was moved to dead letter queue"
    )


class ModelStateResult(BaseModel):
    """Result for get state operations."""

    current_state: EnumCircuitBreakerState = Field(
        description="Current circuit breaker state"
    )

    failure_count: int = Field(
        ge=0,
        description="Current failure count"
    )

    success_count: int = Field(
        ge=0,
        description="Current success count"
    )

    last_failure_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last failure"
    )

    time_in_current_state_seconds: float = Field(
        ge=0.0,
        description="How long in current state (seconds)"
    )

    next_state_transition_estimate: Optional[datetime] = Field(
        default=None,
        description="Estimated time of next state transition"
    )


class ModelResetResult(BaseModel):
    """Result for reset circuit operations."""

    reset_successful: bool = Field(
        description="Whether the reset was successful"
    )

    previous_state: EnumCircuitBreakerState = Field(
        description="Circuit breaker state before reset"
    )

    new_state: EnumCircuitBreakerState = Field(
        description="Circuit breaker state after reset"
    )

    metrics_reset: bool = Field(
        description="Whether metrics were also reset"
    )

    events_cleared_from_queue: int = Field(
        ge=0,
        description="Number of events cleared from queue"
    )

    dead_letter_queue_cleared: bool = Field(
        description="Whether dead letter queue was cleared"
    )


class ModelHealthStatusResult(BaseModel):
    """Result for health status operations."""

    is_healthy: bool = Field(
        description="Whether the circuit breaker is healthy"
    )

    health_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Health score (0-100)"
    )

    circuit_availability_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Circuit availability percentage"
    )

    avg_response_time_ms: float = Field(
        ge=0.0,
        description="Average response time in milliseconds"
    )

    error_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Current error rate percentage"
    )

    queue_utilization_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Queue utilization percentage"
    )

    uptime_seconds: float = Field(
        ge=0.0,
        description="Circuit breaker uptime in seconds"
    )

    issues_detected: List[str] = Field(
        default_factory=list,
        max_items=20,
        description="List of issues detected with the circuit breaker"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        max_items=10,
        description="Health improvement recommendations"
    )

    last_health_check: datetime = Field(
        description="Timestamp of last health check"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
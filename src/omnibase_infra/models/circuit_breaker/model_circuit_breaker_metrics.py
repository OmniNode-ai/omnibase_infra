"""Circuit Breaker Metrics Model.

Shared model for circuit breaker metrics and performance data.
Used across circuit breaker nodes and observability systems.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ModelCircuitBreakerMetrics(BaseModel):
    """Model for circuit breaker metrics tracking."""
    
    total_events: int = Field(
        default=0,
        ge=0,
        description="Total number of events processed"
    )
    
    successful_events: int = Field(
        default=0,
        ge=0,
        description="Number of successfully processed events"
    )
    
    failed_events: int = Field(
        default=0,
        ge=0,
        description="Number of failed events"
    )
    
    queued_events: int = Field(
        default=0,
        ge=0,
        description="Number of events currently queued"
    )
    
    dropped_events: int = Field(
        default=0,
        ge=0,
        description="Number of events dropped due to capacity limits"
    )
    
    dead_letter_events: int = Field(
        default=0,
        ge=0,
        description="Number of events in dead letter queue"
    )
    
    circuit_opens: int = Field(
        default=0,
        ge=0,
        description="Number of times circuit has opened"
    )
    
    circuit_closes: int = Field(
        default=0,
        ge=0,
        description="Number of times circuit has closed"
    )
    
    last_failure: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last failure"
    )
    
    last_success: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last success"
    )
    
    success_rate_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Success rate percentage"
    )
    
    average_response_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average response time in milliseconds"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
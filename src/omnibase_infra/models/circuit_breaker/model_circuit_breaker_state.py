"""Circuit Breaker State Model.

Shared model for circuit breaker state information.
Used across circuit breaker nodes and monitoring systems.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class CircuitBreakerStateEnum(str, Enum):
    """Circuit breaker states for event publishing reliability."""
    CLOSED = "closed"       # Normal operation - events published directly
    OPEN = "open"          # Failure state - events queued or dropped based on policy  
    HALF_OPEN = "half_open"  # Testing state - limited event publishing to test recovery


class ModelCircuitBreakerState(BaseModel):
    """Model for circuit breaker state information."""
    
    state: CircuitBreakerStateEnum = Field(
        description="Current circuit breaker state"
    )
    
    failure_count: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive failures"
    )
    
    success_count: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive successes (in half-open state)"
    )
    
    last_failure_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last failure"
    )
    
    last_success_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last success"
    )
    
    last_state_change: datetime = Field(
        description="Timestamp of last state change"
    )
    
    is_healthy: bool = Field(
        description="Whether circuit breaker is healthy for operations"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
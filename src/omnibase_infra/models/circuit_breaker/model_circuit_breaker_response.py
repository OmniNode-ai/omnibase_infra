"""Circuit Breaker Response Model.

Shared model for circuit breaker operation responses.
Used for returning results from circuit breaker operations.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from uuid import UUID
from datetime import datetime
from .model_circuit_breaker_state import ModelCircuitBreakerState
from .model_circuit_breaker_metrics import ModelCircuitBreakerMetrics


class ModelCircuitBreakerResponse(BaseModel):
    """Model for circuit breaker operation responses."""
    
    operation_type: str = Field(
        description="Type of operation that was executed"
    )
    
    success: bool = Field(
        description="Whether the operation was successful"
    )
    
    correlation_id: UUID = Field(
        description="Request correlation ID for tracing"
    )
    
    timestamp: datetime = Field(
        description="Response timestamp"
    )
    
    execution_time_ms: float = Field(
        ge=0.0,
        description="Operation execution time in milliseconds"
    )
    
    event_published: Optional[bool] = Field(
        default=None,
        description="Whether event was successfully published (for publish_event operations)"
    )
    
    event_queued: Optional[bool] = Field(
        default=None,
        description="Whether event was queued (for publish_event operations)"
    )
    
    circuit_state: Optional[ModelCircuitBreakerState] = Field(
        default=None,
        description="Current circuit breaker state (for check_state operations)"
    )
    
    metrics: Optional[ModelCircuitBreakerMetrics] = Field(
        default=None,
        description="Circuit breaker metrics (for get_metrics operations)"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional response context"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
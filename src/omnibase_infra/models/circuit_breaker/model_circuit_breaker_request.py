"""Circuit Breaker Request Model.

Shared model for circuit breaker operation requests.
Used for event publishing and circuit breaker control operations.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from uuid import UUID
from datetime import datetime


class ModelCircuitBreakerRequest(BaseModel):
    """Model for circuit breaker operation requests."""
    
    operation_type: str = Field(
        description="Type of circuit breaker operation",
        regex=r"^(publish_event|check_state|reset_circuit|get_metrics)$"
    )
    
    correlation_id: UUID = Field(
        description="Request correlation ID for tracing"
    )
    
    timestamp: datetime = Field(
        description="Request timestamp"
    )
    
    event_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Event data for publish_event operations"
    )
    
    publisher_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Publisher configuration for event publishing"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional request context"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
"""Distributed Tracing Output Model.

Node-specific output model for the distributed tracing compute node.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class ModelDistributedTracingOutput(BaseModel):
    """Output model for distributed tracing operations."""
    
    success: bool = Field(
        description="Whether the operation succeeded"
    )
    
    operation_type: str = Field(
        description="Type of operation that was performed"
    )
    
    correlation_id: UUID = Field(
        description="Correlation ID from the request"
    )
    
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Operation-specific result data"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed"
    )
    
    trace_id: Optional[str] = Field(
        default=None,
        description="OpenTelemetry trace ID (if applicable)"
    )
    
    span_id: Optional[str] = Field(
        default=None,
        description="OpenTelemetry span ID (if applicable)"
    )
    
    tracing_enabled: bool = Field(
        description="Whether tracing is currently enabled"
    )
    
    timestamp: datetime = Field(
        description="Response timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
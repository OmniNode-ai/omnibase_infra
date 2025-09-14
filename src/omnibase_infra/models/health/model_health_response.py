"""Health Response Model.

Shared model for health monitoring operation responses.
Used for returning results from health monitoring operations.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
from .model_health_status import ModelHealthStatus
from .model_health_metrics import ModelHealthMetrics
from .model_trend_analysis import ModelTrendAnalysis
from .model_component_status import ModelComponentHealthStatus
from .model_health_alert import ModelHealthAlert


class ModelHealthResponse(BaseModel):
    """Model for health monitoring operation responses."""
    
    operation_type: str = Field(
        description="Type of operation that was executed"
    )
    
    success: bool = Field(
        description="Whether the operation was successful"
    )
    
    correlation_id: UUID = Field(
        description="Request correlation ID for tracking"
    )
    
    timestamp: datetime = Field(
        description="Response timestamp"
    )
    
    execution_time_ms: float = Field(
        ge=0.0,
        description="Operation execution time in milliseconds"
    )
    
    health_status: Optional[ModelHealthStatus] = Field(
        default=None,
        description="Current health status (for health_check operations)"
    )
    
    health_metrics: Optional[ModelHealthMetrics] = Field(
        default=None,
        description="Detailed health metrics (for get_metrics operations)"
    )
    
    trend_analysis: Optional[ModelTrendAnalysis] = Field(
        default=None,
        description="Health trend analysis (for get_trends operations)"
    )

    monitoring_started: Optional[bool] = Field(
        default=None,
        description="Whether monitoring was started (for start_monitoring operations)"
    )

    monitoring_stopped: Optional[bool] = Field(
        default=None,
        description="Whether monitoring was stopped (for stop_monitoring operations)"
    )

    component_statuses: Optional[Dict[str, ModelComponentHealthStatus]] = Field(
        default=None,
        description="Individual component health statuses mapped by component name"
    )

    alerts: Optional[List[ModelHealthAlert]] = Field(
        default=None,
        description="Active health alerts"
    )
    
    prometheus_metrics: Optional[str] = Field(
        default=None,
        description="Prometheus-formatted metrics string"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
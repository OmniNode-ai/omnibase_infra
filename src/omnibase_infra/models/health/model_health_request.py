"""Health Request Model.

Shared model for health monitoring operation requests.
Used for health checks and monitoring control operations.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime


class ModelHealthRequest(BaseModel):
    """Model for health monitoring operation requests."""
    
    operation_type: str = Field(
        description="Type of health monitoring operation",
        regex=r"^(health_check|get_metrics|get_trends|start_monitoring|stop_monitoring)$"
    )
    
    correlation_id: UUID = Field(
        description="Request correlation ID for tracking"
    )
    
    timestamp: datetime = Field(
        description="Request timestamp"
    )
    
    component_filters: Optional[List[str]] = Field(
        default=None,
        description="Filter health checks to specific components (postgres, kafka, circuit_breaker, consul, vault)"
    )
    
    include_metrics: bool = Field(
        default=True,
        description="Include detailed metrics in health response"
    )
    
    include_trends: bool = Field(
        default=False,
        description="Include trend analysis in health response"
    )
    
    trend_hours: Optional[int] = Field(
        default=1,
        gt=0,
        description="Number of hours for trend analysis"
    )
    
    monitoring_interval_seconds: Optional[int] = Field(
        default=30,
        gt=0,
        description="Monitoring interval for start_monitoring operations"
    )
    
    environment: Optional[str] = Field(
        default=None,
        description="Target environment for health checks"
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
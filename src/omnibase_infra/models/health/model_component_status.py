"""Component Status Model.

Strongly-typed model for individual component health statuses.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ModelComponentHealthStatus(BaseModel):
    """Model for individual component health status."""

    component_name: str = Field(
        description="Name of the component"
    )

    status: str = Field(
        pattern="^(healthy|warning|critical|unknown|offline)$",
        description="Current health status of the component"
    )

    last_check_timestamp: datetime = Field(
        description="Timestamp of last health check"
    )

    # Health indicators
    is_available: bool = Field(
        description="Whether the component is available for requests"
    )

    response_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Average response time in milliseconds"
    )

    error_rate_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Error rate percentage"
    )

    # Resource metrics
    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage"
    )

    memory_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Memory usage percentage"
    )

    disk_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Disk usage percentage"
    )

    # Connection metrics
    active_connections: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of active connections"
    )

    connection_pool_utilization: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Connection pool utilization percentage"
    )

    # Performance indicators
    throughput_per_second: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Operations or requests processed per second"
    )

    queue_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Length of processing queue"
    )

    # Health check details
    health_check_duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time taken to complete health check in milliseconds"
    )

    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive health check failures"
    )

    consecutive_successes: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive health check successes"
    )

    # Status details
    status_message: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Detailed status message or error description"
    )

    recovery_actions_available: bool = Field(
        default=False,
        description="Whether automatic recovery actions are available"
    )

    requires_manual_intervention: bool = Field(
        default=False,
        description="Whether manual intervention is required"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
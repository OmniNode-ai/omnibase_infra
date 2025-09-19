"""Health Status Result Model."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelHealthStatusResult(BaseModel):
    """Result for health status operations."""

    is_healthy: bool = Field(
        description="Whether the circuit breaker is healthy",
    )

    health_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Health score (0-100)",
    )

    circuit_availability_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Circuit availability percentage",
    )

    avg_response_time_ms: float = Field(
        ge=0.0,
        description="Average response time in milliseconds",
    )

    error_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Current error rate percentage",
    )

    queue_utilization_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Queue utilization percentage",
    )

    uptime_seconds: float = Field(
        ge=0.0,
        description="Circuit breaker uptime in seconds",
    )

    issues_detected: list[str] = Field(
        default_factory=list,
        max_items=20,
        description="List of issues detected with the circuit breaker",
    )

    recommendations: list[str] = Field(
        default_factory=list,
        max_items=10,
        description="Health improvement recommendations",
    )

    last_health_check: datetime = Field(
        description="Timestamp of last health check",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

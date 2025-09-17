"""Health Status Model.

Shared model for infrastructure health status information.
Used across health monitoring nodes and status reporting.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from omnibase_infra.models.core.health.model_health_details import ModelHealthDetails


class HealthStatusEnum(str, Enum):
    """Infrastructure health status levels."""
    HEALTHY = "healthy"          # All systems operational
    DEGRADED = "degraded"        # Some issues but service available
    UNHEALTHY = "unhealthy"      # Critical issues affecting service


class ModelHealthStatus(BaseModel):
    """Model for infrastructure health status."""

    overall_status: HealthStatusEnum = Field(
        description="Overall infrastructure health status",
    )

    timestamp: datetime = Field(
        description="Health check timestamp",
    )

    environment: str = Field(
        description="Environment where health check was performed",
    )

    service_name: str = Field(
        default="omnibase_infrastructure",
        description="Name of the service being monitored",
    )

    postgres_healthy: bool = Field(
        description="PostgreSQL component health status",
    )

    kafka_healthy: bool = Field(
        description="Kafka component health status",
    )

    circuit_breaker_healthy: bool = Field(
        description="Circuit breaker component health status",
    )

    consul_healthy: bool | None = Field(
        default=None,
        description="Consul service discovery health status",
    )

    vault_healthy: bool | None = Field(
        default=None,
        description="Vault secret management health status",
    )

    health_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall health score (0-100)",
    )

    error_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Current error rate percentage",
    )

    response_time_ms: float = Field(
        ge=0.0,
        description="Average response time in milliseconds",
    )

    uptime_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Service uptime in seconds",
    )

    details: ModelHealthDetails | None = Field(
        default=None,
        description="Additional health status details",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

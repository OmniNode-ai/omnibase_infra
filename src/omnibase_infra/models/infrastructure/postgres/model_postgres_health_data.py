"""Strongly typed PostgreSQL health data model."""

from pydantic import BaseModel, Field

from .model_postgres_connection_pool_health import ModelPostgresConnectionPoolHealth
from .model_postgres_database_health import ModelPostgresDatabaseHealth


class ModelPostgresHealthData(BaseModel):
    """Strongly typed PostgreSQL health data for event publishing."""

    overall_status: str = Field(
        description="Overall health status (healthy, degraded, unhealthy)",
    )

    connection_pool: ModelPostgresConnectionPoolHealth | None = Field(
        default=None,
        description="Connection pool health metrics",
    )

    database: ModelPostgresDatabaseHealth | None = Field(
        default=None,
        description="Database server health metrics",
    )

    response_time_ms: float = Field(
        description="Health check response time in milliseconds",
        ge=0,
    )

    check_timestamp: str = Field(
        description="ISO timestamp when health check was performed",
    )

    error_messages: list[str] = Field(
        default_factory=list,
        description="List of error messages if health issues detected",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="List of warning messages",
    )

    circuit_breaker_state: str = Field(
        description="Circuit breaker state (CLOSED, OPEN, HALF_OPEN)",
    )

    last_failure_time: str | None = Field(
        default=None,
        description="ISO timestamp of last failure (if any)",
    )

"""Strongly typed PostgreSQL health data model."""


from pydantic import BaseModel, Field


class ModelPostgresConnectionPoolHealth(BaseModel):
    """Connection pool health metrics."""

    total_connections: int = Field(
        description="Total number of connections in the pool",
        ge=0,
    )

    active_connections: int = Field(
        description="Number of active connections",
        ge=0,
    )

    idle_connections: int = Field(
        description="Number of idle connections",
        ge=0,
    )

    max_connections: int = Field(
        description="Maximum allowed connections",
        ge=1,
    )

    connection_utilization_percent: float = Field(
        description="Connection utilization as percentage",
        ge=0.0,
        le=100.0,
    )


class ModelPostgresDatabaseHealth(BaseModel):
    """Database server health metrics."""

    server_version: str = Field(
        description="PostgreSQL server version",
    )

    database_name: str = Field(
        description="Name of the connected database",
    )

    is_read_only: bool = Field(
        description="Whether the database is in read-only mode",
    )

    uptime_seconds: int | None = Field(
        default=None,
        description="Server uptime in seconds",
        ge=0,
    )

    total_size_bytes: int | None = Field(
        default=None,
        description="Total database size in bytes",
        ge=0,
    )

    lock_count: int | None = Field(
        default=None,
        description="Number of active locks",
        ge=0,
    )


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

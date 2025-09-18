"""PostgreSQL connection pool health model."""

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
"""PostgreSQL connection statistics model."""

from pydantic import BaseModel, Field


class ModelPostgresConnectionStats(BaseModel):
    """
    Connection pool statistics for monitoring and observability.

    Tracks connection usage, performance metrics, and health indicators
    for PostgreSQL connection pools.
    """

    size: int = Field(default=0, description="Current pool size", ge=0)
    checked_out: int = Field(default=0, description="Connections currently checked out", ge=0)
    overflow: int = Field(default=0, description="Overflow connections", ge=0)
    checked_in: int = Field(default=0, description="Connections checked back in", ge=0)
    total_connections: int = Field(default=0, description="Total connections created", ge=0)
    failed_connections: int = Field(default=0, description="Number of failed connection attempts", ge=0)
    reconnect_count: int = Field(default=0, description="Number of reconnections", ge=0)
    query_count: int = Field(default=0, description="Total queries executed", ge=0)
    average_response_time_ms: float = Field(
        default=0.0,
        description="Average query response time in milliseconds",
        ge=0.0,
    )

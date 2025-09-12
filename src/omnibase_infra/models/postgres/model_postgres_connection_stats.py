"""PostgreSQL connection statistics model."""

from pydantic import BaseModel, Field


class ModelPostgresConnectionStats(BaseModel):
    """Connection pool statistics."""

    size: int = Field(description="Current pool size")
    checked_out: int = Field(description="Connections currently checked out")
    overflow: int = Field(description="Overflow connections")
    checked_in: int = Field(description="Connections checked back in")
    total_connections: int = Field(description="Total connections created")
    failed_connections: int = Field(description="Number of failed connection attempts")
    reconnect_count: int = Field(description="Number of reconnections")
    query_count: int = Field(description="Total queries executed")
    average_response_time_ms: float = Field(description="Average query response time in milliseconds")
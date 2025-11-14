"""PostgreSQL connection pool information model."""


from pydantic import BaseModel, Field


class ModelPostgresConnectionPoolInfo(BaseModel):
    """PostgreSQL connection pool information model."""

    total_connections: int = Field(description="Total number of connections in pool", ge=0)
    active_connections: int = Field(description="Number of active connections", ge=0)
    idle_connections: int = Field(description="Number of idle connections", ge=0)
    pool_size_limit: int = Field(description="Maximum pool size", ge=1)
    pool_name: str | None = Field(default=None, description="Name of the connection pool")
    average_connection_time_ms: float | None = Field(default=None, description="Average connection time in milliseconds", ge=0)
    pool_health: str = Field(default="healthy", description="Pool health status: healthy, degraded, unhealthy")

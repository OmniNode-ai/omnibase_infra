"""PostgreSQL database health model."""

from pydantic import BaseModel, Field


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
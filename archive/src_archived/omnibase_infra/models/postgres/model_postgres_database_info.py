"""PostgreSQL database information model."""

from pydantic import BaseModel, Field


class ModelPostgresDatabaseInfo(BaseModel):
    """PostgreSQL database information model."""

    database_name: str = Field(description="Name of the database")
    database_version: str = Field(description="PostgreSQL version")
    database_size_bytes: int | None = Field(
        default=None, description="Database size in bytes", ge=0,
    )
    connection_count: int = Field(
        description="Current number of database connections", ge=0,
    )
    max_connections: int = Field(description="Maximum allowed connections", ge=1)
    uptime_seconds: int | None = Field(
        default=None, description="Database uptime in seconds", ge=0,
    )
    is_read_only: bool = Field(
        default=False, description="Whether database is in read-only mode",
    )

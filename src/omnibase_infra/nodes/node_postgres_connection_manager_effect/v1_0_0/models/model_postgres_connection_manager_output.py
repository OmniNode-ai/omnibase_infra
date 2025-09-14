"""Output model for PostgreSQL connection manager EFFECT node."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelPostgresConnectionManagerOutput(BaseModel):
    """Output model for PostgreSQL connection manager operations."""

    success: bool = Field(
        description="Whether the operation completed successfully"
    )
    correlation_id: UUID = Field(
        description="Unique identifier for request tracing"
    )
    timestamp: datetime = Field(
        description="Timestamp of operation completion"
    )
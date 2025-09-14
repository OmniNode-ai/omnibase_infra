"""Get health output model for PostgreSQL EFFECT node."""

from datetime import datetime

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_health_data import ModelPostgresHealthData


class ModelGetHealthOutput(BaseModel):
    """Output model for get_health operation."""

    success: bool = Field(
        description="Whether health check completed successfully"
    )
    health_data: ModelPostgresHealthData = Field(
        description="Comprehensive PostgreSQL health data"
    )
    timestamp: datetime = Field(
        description="Health check timestamp"
    )
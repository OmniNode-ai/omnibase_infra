"""PostgreSQL health check request model."""

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.models.infrastructure.postgres.model_postgres_context import ModelPostgresContext


class ModelPostgresHealthRequest(BaseModel):
    """PostgreSQL health check request model."""

    include_performance_metrics: bool = Field(default=True, description="Include performance metrics in response")
    include_connection_stats: bool = Field(default=True, description="Include connection pool statistics")
    include_schema_info: bool = Field(default=True, description="Include schema validation information")
    correlation_id: UUID | None = Field(default=None, description="Request correlation ID")
    context: ModelPostgresContext | None = Field(default=None, description="Additional request context")

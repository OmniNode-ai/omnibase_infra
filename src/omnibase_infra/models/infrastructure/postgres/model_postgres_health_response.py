"""PostgreSQL health check response model."""

from uuid import UUID

from pydantic import BaseModel, Field

from .model_postgres_connection_pool_info import ModelPostgresConnectionPoolInfo
from .model_postgres_context import ModelPostgresContext
from .model_postgres_database_info import ModelPostgresDatabaseInfo
from .model_postgres_error import ModelPostgresError
from .model_postgres_performance_metrics import ModelPostgresPerformanceMetrics
from .model_postgres_schema_info import ModelPostgresSchemaInfo


class ModelPostgresHealthResponse(BaseModel):
    """PostgreSQL health check response model."""

    status: str = Field(description="Health status: healthy, degraded, unhealthy")
    timestamp: float = Field(description="Health check timestamp")
    connection_pool: ModelPostgresConnectionPoolInfo | None = Field(
        default=None, description="Connection pool information",
    )
    database_info: ModelPostgresDatabaseInfo | None = Field(
        default=None, description="Database information",
    )
    schema_info: ModelPostgresSchemaInfo | None = Field(
        default=None, description="Schema validation information",
    )
    performance: ModelPostgresPerformanceMetrics | None = Field(
        default=None, description="Performance metrics",
    )
    errors: list[ModelPostgresError] = Field(default_factory=list, description="List of errors or warnings")
    correlation_id: UUID | None = Field(default=None, description="Request correlation ID")
    context: ModelPostgresContext | None = Field(default=None, description="Additional response context")

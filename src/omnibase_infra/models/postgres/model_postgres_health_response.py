"""PostgreSQL health check response model."""

from typing import List, Optional
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
    connection_pool: Optional[ModelPostgresConnectionPoolInfo] = Field(
        default=None, description="Connection pool information"
    )
    database_info: Optional[ModelPostgresDatabaseInfo] = Field(
        default=None, description="Database information"
    )
    schema_info: Optional[ModelPostgresSchemaInfo] = Field(
        default=None, description="Schema validation information"
    )
    performance: Optional[ModelPostgresPerformanceMetrics] = Field(
        default=None, description="Performance metrics"
    )
    errors: List[ModelPostgresError] = Field(default_factory=list, description="List of errors or warnings")
    correlation_id: Optional[UUID] = Field(default=None, description="Request correlation ID")
    context: Optional[ModelPostgresContext] = Field(default=None, description="Additional response context")
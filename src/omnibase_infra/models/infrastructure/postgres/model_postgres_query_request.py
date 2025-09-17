"""PostgreSQL query request model for message bus integration."""

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.models.infrastructure.postgres.enum_postgres_query_type import EnumPostgresQueryType
from omnibase_infra.models.infrastructure.postgres.model_postgres_context import ModelPostgresContext
from omnibase_infra.models.infrastructure.postgres.model_postgres_query_parameter import ModelPostgresQueryParameters


class ModelPostgresQueryRequest(BaseModel):
    """PostgreSQL query request model."""

    query: str = Field(description="SQL query to execute")
    parameters: ModelPostgresQueryParameters = Field(
        default_factory=ModelPostgresQueryParameters,
        description="Query parameters with strongly typed structure",
    )
    timeout: float | None = Field(default=None, description="Query timeout in seconds")
    record_metrics: bool = Field(default=True, description="Whether to record query metrics")
    query_type: EnumPostgresQueryType = Field(default=EnumPostgresQueryType.GENERAL, description="Type of query")
    correlation_id: UUID | None = Field(default=None, description="Request correlation ID")
    context: ModelPostgresContext | None = Field(default=None, description="Additional request context")

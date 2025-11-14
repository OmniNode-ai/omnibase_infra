"""PostgreSQL query response model for message bus integration."""

from uuid import UUID

from pydantic import BaseModel, Field

from .model_postgres_context import ModelPostgresContext
from .model_postgres_error import ModelPostgresError
from .model_postgres_query_metrics import ModelPostgresQueryMetrics
from .model_postgres_query_result import ModelPostgresQueryResult


class ModelPostgresQueryResponse(BaseModel):
    """PostgreSQL query response model."""

    success: bool = Field(description="Whether the query was successful")
    data: ModelPostgresQueryResult | None = Field(default=None, description="Query result data")
    status_message: str | None = Field(default=None, description="Database status message")
    rows_affected: int = Field(default=0, description="Number of rows affected/returned")
    execution_time_ms: float = Field(description="Query execution time in milliseconds")
    correlation_id: UUID | None = Field(default=None, description="Request correlation ID")
    error: ModelPostgresError | None = Field(default=None, description="Error details if query failed")
    query_metrics: ModelPostgresQueryMetrics | None = Field(default=None, description="Detailed query metrics")
    context: ModelPostgresContext | None = Field(default=None, description="Additional response context")

"""PostgreSQL query response model for message bus integration."""

from typing import Optional

from pydantic import BaseModel, Field

from .model_postgres_context import ModelPostgresContext
from .model_postgres_error import ModelPostgresError
from .model_postgres_query_metrics import ModelPostgresQueryMetrics
from .model_postgres_query_result import ModelPostgresQueryResult


class ModelPostgresQueryResponse(BaseModel):
    """PostgreSQL query response model."""

    success: bool = Field(description="Whether the query was successful")
    data: Optional[ModelPostgresQueryResult] = Field(default=None, description="Query result data")
    status_message: Optional[str] = Field(default=None, description="Database status message")
    rows_affected: int = Field(default=0, description="Number of rows affected/returned")
    execution_time_ms: float = Field(description="Query execution time in milliseconds")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    error: Optional[ModelPostgresError] = Field(default=None, description="Error details if query failed")
    query_metrics: Optional[ModelPostgresQueryMetrics] = Field(default=None, description="Detailed query metrics")
    context: Optional[ModelPostgresContext] = Field(default=None, description="Additional response context")
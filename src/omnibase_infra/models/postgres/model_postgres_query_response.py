"""PostgreSQL query response model for message bus integration."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_query_metrics import ModelPostgresQueryMetrics


class ModelPostgresQueryResponse(BaseModel):
    """PostgreSQL query response model."""

    success: bool = Field(description="Whether the query was successful")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Query result data")
    status_message: Optional[str] = Field(default=None, description="Database status message")
    rows_affected: int = Field(default=0, description="Number of rows affected/returned")
    execution_time_ms: float = Field(description="Query execution time in milliseconds")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    error_message: Optional[str] = Field(default=None, description="Error message if query failed")
    query_metrics: Optional[ModelPostgresQueryMetrics] = Field(default=None, description="Detailed query metrics")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional response context")
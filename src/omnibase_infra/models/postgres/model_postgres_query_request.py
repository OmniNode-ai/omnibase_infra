"""PostgreSQL query request model for message bus integration."""

from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .enum_postgres_query_type import EnumPostgresQueryType
from .model_postgres_context import ModelPostgresContext


class ModelPostgresQueryRequest(BaseModel):
    """PostgreSQL query request model."""

    query: str = Field(description="SQL query to execute")
    parameters: List[Any] = Field(default_factory=list, description="Query parameters")
    timeout: Optional[float] = Field(default=None, description="Query timeout in seconds")
    record_metrics: bool = Field(default=True, description="Whether to record query metrics")
    query_type: EnumPostgresQueryType = Field(default=EnumPostgresQueryType.GENERAL, description="Type of query")
    correlation_id: Optional[UUID] = Field(default=None, description="Request correlation ID")
    context: Optional[ModelPostgresContext] = Field(default=None, description="Additional request context")
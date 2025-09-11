"""PostgreSQL adapter input envelope model."""

from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.postgres.model_postgres_health_request import ModelPostgresHealthRequest
from omnibase_infra.models.postgres.model_postgres_context import ModelPostgresContext
from .enum_postgres_operation_type import EnumPostgresOperationType


class ModelPostgresAdapterInput(BaseModel):
    """Input envelope for PostgreSQL adapter operations."""

    operation_type: EnumPostgresOperationType = Field(description="Type of operation")
    
    query_request: Optional[ModelPostgresQueryRequest] = Field(
        default=None, description="Query request payload (when operation_type is 'query')"
    )
    
    health_request: Optional[ModelPostgresHealthRequest] = Field(
        default=None, description="Health check request payload (when operation_type is 'health_check')"
    )
    
    correlation_id: str = Field(description="Request correlation ID for tracing")
    
    timestamp: datetime = Field(description="Request timestamp")
    
    context: Optional[ModelPostgresContext] = Field(
        default=None, description="Additional request context"
    )
"""PostgreSQL adapter output envelope model."""

from typing import Optional
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_query_response import ModelPostgresQueryResponse
from omnibase_infra.models.postgres.model_postgres_health_response import ModelPostgresHealthResponse
from omnibase_infra.models.postgres.model_postgres_context import ModelPostgresContext
from omnibase_infra.models.postgres.model_postgres_error import ModelPostgresError
from ..enums.enum_postgres_operation_type import EnumPostgresOperationType


class ModelPostgresAdapterOutput(BaseModel):
    """Output envelope for PostgreSQL adapter operations."""

    operation_type: EnumPostgresOperationType = Field(description="Type of operation that was executed")
    
    query_response: Optional[ModelPostgresQueryResponse] = Field(
        default=None, description="Query response payload (when operation_type is 'query')"
    )
    
    health_response: Optional[ModelPostgresHealthResponse] = Field(
        default=None, description="Health check response payload (when operation_type is 'health_check')"
    )
    
    success: bool = Field(description="Whether the operation was successful")
    
    error: Optional[ModelPostgresError] = Field(
        default=None, description="Error details if operation failed"
    )
    
    correlation_id: UUID = Field(description="Request correlation ID for tracing")
    
    timestamp: datetime = Field(description="Response timestamp")
    
    execution_time_ms: float = Field(description="Total operation execution time in milliseconds")
    
    context: Optional[ModelPostgresContext] = Field(
        default=None, description="Additional response context"
    )
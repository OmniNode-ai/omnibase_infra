"""PostgreSQL adapter output envelope model."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_query_response import ModelPostgresQueryResponse
from omnibase_infra.models.postgres.model_postgres_health_response import ModelPostgresHealthResponse


class ModelPostgresAdapterOutput(BaseModel):
    """Output envelope for PostgreSQL adapter operations."""

    operation_type: str = Field(description="Type of operation that was executed")
    
    query_response: Optional[ModelPostgresQueryResponse] = Field(
        default=None, description="Query response payload (when operation_type is 'query')"
    )
    
    health_response: Optional[ModelPostgresHealthResponse] = Field(
        default=None, description="Health check response payload (when operation_type is 'health_check')"
    )
    
    success: bool = Field(description="Whether the operation was successful")
    
    error_message: Optional[str] = Field(
        default=None, description="Error message if operation failed"
    )
    
    correlation_id: str = Field(description="Request correlation ID for tracing")
    
    timestamp: float = Field(description="Response timestamp")
    
    execution_time_ms: float = Field(description="Total operation execution time in milliseconds")
    
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response context"
    )
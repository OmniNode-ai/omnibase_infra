"""PostgreSQL adapter input envelope model."""

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.postgres.model_postgres_health_request import ModelPostgresHealthRequest


class ModelPostgresAdapterInput(BaseModel):
    """Input envelope for PostgreSQL adapter operations."""

    operation_type: str = Field(description="Type of operation: query, health_check")
    
    query_request: Optional[ModelPostgresQueryRequest] = Field(
        default=None, description="Query request payload (when operation_type is 'query')"
    )
    
    health_request: Optional[ModelPostgresHealthRequest] = Field(
        default=None, description="Health check request payload (when operation_type is 'health_check')"
    )
    
    correlation_id: UUID = Field(description="Request correlation ID for tracing")
    
    timestamp: float = Field(description="Request timestamp")
    
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional request context"
    )
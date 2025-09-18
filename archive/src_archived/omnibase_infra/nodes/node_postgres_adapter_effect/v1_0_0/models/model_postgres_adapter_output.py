"""PostgreSQL adapter output envelope model."""

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_context import ModelPostgresContext
from omnibase_infra.models.postgres.model_postgres_health_response import (
    ModelPostgresHealthResponse,
)
from omnibase_infra.models.postgres.model_postgres_query_response import (
    ModelPostgresQueryResponse,
)

from ..enums.enum_postgres_operation_type import EnumPostgresOperationType


class ModelPostgresAdapterOutput(BaseModel):
    """Output envelope for PostgreSQL adapter operations."""

    operation_type: EnumPostgresOperationType = Field(
        description="Type of operation that was executed",
    )

    query_response: ModelPostgresQueryResponse | None = Field(
        default=None,
        description="Query response payload (when operation_type is 'query')",
    )

    health_response: ModelPostgresHealthResponse | None = Field(
        default=None,
        description="Health check response payload (when operation_type is 'health_check')",
    )

    success: bool = Field(description="Whether the operation was successful")

    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )

    correlation_id: UUID = Field(description="Request correlation ID for tracing")

    timestamp: float = Field(description="Response timestamp as Unix timestamp", ge=0)

    execution_time_ms: float = Field(
        description="Total operation execution time in milliseconds", ge=0,
    )

    context: ModelPostgresContext | None = Field(
        default=None,
        description="Additional response context",
    )

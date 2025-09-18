"""PostgreSQL adapter input envelope model."""

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.models.postgres.model_postgres_context import ModelPostgresContext
from omnibase_infra.models.postgres.model_postgres_health_request import (
    ModelPostgresHealthRequest,
)
from omnibase_infra.models.postgres.model_postgres_query_request import (
    ModelPostgresQueryRequest,
)

from ..enums.enum_postgres_operation_type import EnumPostgresOperationType


class ModelPostgresAdapterInput(BaseModel):
    """Input envelope for PostgreSQL adapter operations."""

    operation_type: EnumPostgresOperationType = Field(description="Type of operation")

    query_request: ModelPostgresQueryRequest | None = Field(
        default=None,
        description="Query request payload (when operation_type is 'query')",
    )

    health_request: ModelPostgresHealthRequest | None = Field(
        default=None,
        description="Health check request payload (when operation_type is 'health_check')",
    )

    correlation_id: UUID = Field(description="Request correlation ID for tracing")

    timestamp: float = Field(description="Request timestamp as Unix timestamp", ge=0)

    context: ModelPostgresContext | None = Field(
        default=None,
        description="Additional request context",
    )

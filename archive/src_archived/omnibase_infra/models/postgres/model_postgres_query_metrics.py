"""PostgreSQL query execution metrics model."""

from datetime import datetime

from pydantic import BaseModel, Field

from .model_postgres_connection_id import ModelPostgresConnectionId
from .model_postgres_error import ModelPostgresError


class ModelPostgresQueryMetrics(BaseModel):
    """Query execution metrics."""

    query_hash: str = Field(description="Hash of the executed query")
    execution_time_ms: float = Field(description="Query execution time in milliseconds")
    rows_affected: int = Field(description="Number of rows affected/returned")
    connection_info: ModelPostgresConnectionId = Field(
        description="Connection information",
    )
    timestamp: datetime = Field(description="Timestamp of query execution")
    was_successful: bool = Field(description="Whether query executed successfully")
    error: ModelPostgresError | None = Field(
        default=None, description="Error details if query failed",
    )

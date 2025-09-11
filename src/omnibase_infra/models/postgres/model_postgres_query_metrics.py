"""PostgreSQL query execution metrics model."""

from typing import Optional

from pydantic import BaseModel, Field


class ModelPostgresQueryMetrics(BaseModel):
    """Query execution metrics."""

    query_hash: str = Field(description="Hash of the executed query")
    execution_time_ms: float = Field(description="Query execution time in milliseconds")
    rows_affected: int = Field(description="Number of rows affected/returned")
    connection_id: str = Field(description="Connection identifier")
    timestamp: float = Field(description="Timestamp of query execution")
    was_successful: bool = Field(description="Whether query executed successfully")
    error_message: Optional[str] = Field(default=None, description="Error message if query failed")
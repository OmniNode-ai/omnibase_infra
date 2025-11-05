"""PostgreSQL query metrics model."""

from pydantic import BaseModel, Field


class ModelPostgresQueryMetrics(BaseModel):
    """
    Query execution metrics for performance monitoring and debugging.

    Captures detailed information about individual query execution including
    timing, affected rows, and error information.
    """

    query_hash: str = Field(description="Hash of the query for identification")
    execution_time_ms: float = Field(description="Query execution time in milliseconds", ge=0.0)
    rows_affected: int = Field(description="Number of rows affected or returned", ge=0)
    connection_id: str = Field(description="Connection identifier")
    timestamp: float = Field(description="Unix timestamp of query execution", ge=0.0)
    was_successful: bool = Field(description="Whether the query executed successfully")
    error_message: str | None = Field(default=None, description="Error message if query failed")

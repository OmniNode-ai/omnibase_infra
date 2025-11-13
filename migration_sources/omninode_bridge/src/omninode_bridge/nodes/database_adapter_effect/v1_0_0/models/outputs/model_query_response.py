"""
Query Response Model.

Output model for database query operations that return result sets.
"""

from typing import Any, ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelQueryResponse(BaseModel):
    """
    Response from database query operations.

    This model extends the base operation output to include query result data.
    Used for SELECT queries that return rows from the database.

    Attributes:
        success: Whether the query executed successfully
        correlation_id: UUID preserved from input for end-to-end tracking
        execution_time_ms: Time taken to execute the query in milliseconds
        rows: List of result rows (each row is a dict mapping column name to value)
        row_count: Number of rows returned by the query
        columns: Optional list of column names in result set
        error_message: Human-readable error description if query failed
    """

    success: bool = Field(
        ...,
        description="Query execution success status",
    )

    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID preserved from input",
    )

    execution_time_ms: int = Field(
        ...,
        description="Query execution time in milliseconds",
        ge=0,
    )

    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Query result rows (each row is a dict mapping column to value)",
    )

    row_count: int = Field(
        default=0,
        description="Number of rows returned by the query",
        ge=0,
    )

    columns: Optional[list[str]] = Field(
        default=None,
        description="Column names in result set (if available)",
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if query failed (None on success)",
    )

    class Config:
        """Pydantic v2 configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "examples": [
                {
                    "success": True,
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "execution_time_ms": 12,
                    "rows": [
                        {
                            "bridge_id": "550e8400-e29b-41d4-a716-446655440001",
                            "namespace": "production",
                            "total_workflows_processed": 1523,
                            "current_fsm_state": "ACTIVE",
                        },
                        {
                            "bridge_id": "550e8400-e29b-41d4-a716-446655440002",
                            "namespace": "staging",
                            "total_workflows_processed": 342,
                            "current_fsm_state": "ACTIVE",
                        },
                    ],
                    "row_count": 2,
                    "columns": [
                        "bridge_id",
                        "namespace",
                        "total_workflows_processed",
                        "current_fsm_state",
                    ],
                    "error_message": None,
                },
                {
                    "success": False,
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440003",
                    "execution_time_ms": 8,
                    "rows": [],
                    "row_count": 0,
                    "columns": None,
                    "error_message": "Syntax error in SQL query at position 45",
                },
            ]
        }

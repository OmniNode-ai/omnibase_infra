"""
Database Operation Output Model.

Base output model for all database adapter effect node operations.
Preserves UUID correlation and provides execution metrics.
"""

from typing import Any, ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelDatabaseOperationOutput(BaseModel):
    """
    Output from database adapter effect node operations.

    This model is returned by all database operation handlers and provides
    consistent structure for success/failure reporting, execution metrics,
    and correlation tracking.

    Attributes:
        success: Whether the database operation completed successfully
        operation_type: Type of database operation performed (from EnumDatabaseOperationType)
        correlation_id: UUID preserved from input for end-to-end tracking
        execution_time_ms: Time taken to execute the database operation in milliseconds
        rows_affected: Number of database rows inserted/updated/deleted (0 if query)
        error_message: Human-readable error description if operation failed
    """

    success: bool = Field(
        ...,
        description="Operation success status",
    )

    operation_type: str = Field(
        ...,
        description="Type of database operation performed",
    )

    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID preserved from input",
    )

    execution_time_ms: int = Field(
        ...,
        description="Execution time in milliseconds",
        ge=0,
    )

    rows_affected: int = Field(
        default=0,
        description="Number of rows affected by the operation",
        ge=0,
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed (None on success)",
    )

    result_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Operation result data (e.g., generated IDs, query results)",
    )

    class Config:
        """Pydantic v2 configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "examples": [
                {
                    "success": True,
                    "operation_type": "persist_bridge_state",
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "execution_time_ms": 8,
                    "rows_affected": 1,
                    "error_message": None,
                },
                {
                    "success": False,
                    "operation_type": "persist_workflow_execution",
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
                    "execution_time_ms": 15,
                    "rows_affected": 0,
                    "error_message": "Foreign key constraint violation: workflow_id does not exist",
                },
            ]
        }

"""Workflow step entity model.

Maps to the workflow_steps database table for tracking individual workflow step execution.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelWorkflowStep(BaseModel):
    """
    Workflow step entity model (maps to workflow_steps table).

    Tracks individual steps within workflow executions, providing granular
    execution monitoring and debugging capabilities.

    Attributes:
        id: Auto-generated primary key (UUID)
        workflow_id: Foreign key to parent workflow execution
        step_name: Name of the workflow step
        step_order: Sequential order of step within workflow
        status: Step status (PENDING, RUNNING, COMPLETED, FAILED)
        execution_time_ms: Step execution time in milliseconds
        step_data: Step-specific data as JSONB
        error_message: Error message if step failed
        created_at: Record creation timestamp (auto-managed)

    Example:
        ```python
        from uuid import uuid4

        step = ModelWorkflowStep(
            workflow_id=uuid4(),
            step_name="hash_generation",
            step_order=1,
            status="COMPLETED",
            execution_time_ms=15,
            step_data={"file_size": 1024, "hash_algorithm": "blake3"}
        )
        ```
    """

    # Auto-generated primary key
    id: Optional[UUID] = Field(
        default=None, description="Auto-generated UUID primary key"
    )

    # Required fields
    workflow_id: UUID = Field(
        ..., description="Foreign key to parent workflow execution"
    )
    step_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the workflow step",
        examples=["hash_generation", "stamp_creation", "intelligence_request"],
    )
    step_order: int = Field(
        ..., ge=0, description="Sequential order of step within workflow (0-based)"
    )
    status: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Step execution status",
        examples=["PENDING", "RUNNING", "COMPLETED", "FAILED"],
    )

    # Optional fields
    execution_time_ms: Optional[int] = Field(
        default=None, ge=0, description="Step execution time in milliseconds"
    )
    step_data: dict[str, Any] = Field(
        default_factory=dict, description="Step-specific data and results as JSONB"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if step failed"
    )

    # Auto-managed timestamp
    created_at: Optional[datetime] = Field(
        default=None, description="Record creation timestamp (auto-managed by DB)"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of the allowed step states."""
        allowed_statuses = {"PENDING", "RUNNING", "COMPLETED", "FAILED"}
        if v.upper() not in allowed_statuses:
            raise ValueError(f"status must be one of {allowed_statuses}, got: {v}")
        return v.upper()

    @field_validator("step_order")
    @classmethod
    def validate_step_order(cls, v: int) -> int:
        """Validate step_order is non-negative."""
        if v < 0:
            raise ValueError("step_order must be non-negative")
        return v

    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode for DB row mapping
        json_schema_extra={
            "example": {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
                "step_name": "hash_generation",
                "step_order": 0,
                "status": "COMPLETED",
                "execution_time_ms": 15,
                "step_data": {
                    "file_size_bytes": 1024,
                    "hash_algorithm": "blake3",
                    "hash_value": "abc123...",
                },
            }
        },
    )

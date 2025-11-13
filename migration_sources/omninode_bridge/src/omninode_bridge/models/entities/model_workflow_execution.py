"""Workflow execution entity model.

Maps to the workflow_executions database table for tracking orchestrator workflow execution.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omninode_bridge.schemas import WORKFLOW_METADATA_SCHEMA
from omninode_bridge.utils.metadata_validator import validate_metadata


class ModelWorkflowExecution(BaseModel):
    """
    Workflow execution entity model (maps to workflow_executions table).

    Tracks complete workflow execution state including FSM state, timing,
    and execution metadata for orchestrator workflows.

    Attributes:
        id: Auto-generated primary key (UUID)
        correlation_id: Unique correlation ID for cross-service tracking
        workflow_type: Type of workflow (e.g., "metadata_stamping")
        current_state: Current FSM state (PENDING, PROCESSING, COMPLETED, FAILED)
        namespace: Multi-tenant namespace identifier
        started_at: Workflow start timestamp
        completed_at: Workflow completion timestamp (None if in progress)
        execution_time_ms: Total execution time in milliseconds
        error_message: Error message if workflow failed
        metadata: Additional workflow metadata as JSONB
        created_at: Record creation timestamp (auto-managed)
        updated_at: Last update timestamp (auto-managed)

    Example:
        ```python
        from uuid import uuid4
        from datetime import datetime, timezone

        execution = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="metadata_stamping",
            current_state="PROCESSING",
            namespace="test_app",
            started_at=datetime.now(timezone.utc),
            metadata={"user_id": "user-123", "priority": "high"}
        )
        ```
    """

    # Auto-generated primary key
    id: Optional[UUID] = Field(
        default=None, description="Auto-generated UUID primary key"
    )

    # Required fields
    correlation_id: UUID = Field(
        ..., description="Unique correlation ID for cross-service tracking"
    )
    workflow_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of workflow being executed",
        examples=["metadata_stamping", "intelligence_gathering"],
    )
    current_state: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Current FSM state",
        examples=["PENDING", "PROCESSING", "COMPLETED", "FAILED"],
    )
    namespace: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Multi-tenant namespace identifier",
        examples=["test_app", "production_app"],
    )
    started_at: datetime = Field(..., description="Workflow start timestamp")

    # Optional fields
    completed_at: Optional[datetime] = Field(
        default=None, description="Workflow completion timestamp (None if in progress)"
    )
    execution_time_ms: Optional[int] = Field(
        default=None, ge=0, description="Total execution time in milliseconds"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if workflow failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional workflow metadata as JSONB"
    )

    # Auto-managed timestamps
    created_at: Optional[datetime] = Field(
        default=None, description="Record creation timestamp (auto-managed by DB)"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp (auto-managed by DB)"
    )

    @field_validator("current_state")
    @classmethod
    def validate_current_state(cls, v: str) -> str:
        """Validate current_state is one of the allowed FSM states."""
        allowed_states = {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}
        if v.upper() not in allowed_states:
            raise ValueError(f"current_state must be one of {allowed_states}, got: {v}")
        return v.upper()

    @field_validator("execution_time_ms")
    @classmethod
    def validate_execution_time(cls, v: Optional[int]) -> Optional[int]:
        """Validate execution_time_ms is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("execution_time_ms must be non-negative")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_field(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata against JSON schema."""
        if v:
            validate_metadata(v, WORKFLOW_METADATA_SCHEMA)
        return v

    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode for DB row mapping
        json_schema_extra={
            "example": {
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                "workflow_type": "metadata_stamping",
                "current_state": "COMPLETED",
                "namespace": "test_app",
                "started_at": "2025-10-08T12:00:00Z",
                "completed_at": "2025-10-08T12:05:00Z",
                "execution_time_ms": 5234,
                "metadata": {
                    "user_id": "user-123",
                    "priority": "high",
                    "steps_completed": 3,
                },
            }
        },
    )

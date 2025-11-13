#!/usr/bin/env python3
"""
ModelWorkflowStepInput - Workflow Step History Input.

Input model for persisting individual workflow step execution records.
Tracks detailed step-by-step execution history for workflow debugging
and analytics.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelWorkflowStepInput
- UUID correlation and workflow tracking
- Step status validation
- Comprehensive field validation with Pydantic v2
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkflowStepInput(BaseModel):
    """
    Input model for workflow step database operations.

    This model represents individual steps within a workflow execution.
    Each step is persisted as a separate record in the workflow_steps table,
    enabling detailed execution tracing and performance analysis.

    Database Table: workflow_steps
    Primary Key: id (UUID, auto-generated)
    Foreign Key: workflow_id → workflow_executions(id)
    Indexes: workflow_id, step_order, status

    Step Statuses:
    - PENDING: Step queued, not yet started
    - RUNNING: Step actively executing
    - COMPLETED: Step finished successfully
    - FAILED: Step encountered an error
    - SKIPPED: Step bypassed due to conditional logic

    Event Sources:
    - STEP_COMPLETED: Published by orchestrator after each step
    - HASH_GENERATED: Hash generation step completion
    - STAMP_CREATED: Stamping step completion
    - INTELLIGENCE_RECEIVED: OnexTree analysis step completion

    Example (Hash Generation Step):
        >>> from uuid import uuid4
        >>> from datetime import datetime
        >>> step_data = ModelWorkflowStepInput(
        ...     workflow_id=uuid4(),
        ...     step_name="generate_blake3_hash",
        ...     step_order=1,
        ...     status="COMPLETED",
        ...     execution_time_ms=2,
        ...     step_data={
        ...         "file_hash": "abc123...",
        ...         "file_size_bytes": 1024,
        ...         "performance_grade": "A"
        ...     }
        ... )

    Example (Failed Step):
        >>> failed_step = ModelWorkflowStepInput(
        ...     workflow_id=uuid4(),
        ...     step_name="stamp_content",
        ...     step_order=2,
        ...     status="FAILED",
        ...     execution_time_ms=150,
        ...     error_message="Service timeout after 150ms",
        ...     step_data={"retry_count": 3}
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # === Workflow Association ===
    workflow_id: UUID = Field(
        ...,
        description="""
        UUID of the parent workflow execution.

        References workflow_executions(id) via foreign key.
        All steps for a workflow share the same workflow_id.
        """,
    )

    # === Step Identification ===
    step_name: str = Field(
        ...,
        description="""
        Name/identifier for this workflow step.

        Common step names:
        - "generate_blake3_hash": Hash generation
        - "stamp_content": Content stamping
        - "request_intelligence": OnexTree intelligence request
        - "persist_state": State persistence
        - "publish_event": Event publishing
        - "validate_stamp": Stamp validation
        """,
        min_length=1,
        max_length=100,
    )

    step_order: int = Field(
        ...,
        description="""
        Sequential order of this step within the workflow.

        Step order starts at 1 and increments for each step.
        Enables reconstruction of workflow execution timeline.
        """,
        ge=1,
    )

    # === Step Status ===
    status: str = Field(
        ...,
        description="""
        Execution status of the step.

        Valid values: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED

        Status transitions:
        - PENDING → RUNNING: Step starts
        - RUNNING → COMPLETED: Step succeeds
        - RUNNING → FAILED: Step encounters error
        - PENDING → SKIPPED: Step bypassed by conditional logic
        """,
        min_length=1,
        max_length=50,
    )

    # === Performance Metrics ===
    execution_time_ms: int | None = Field(
        default=None,
        description="""
        Step execution time in milliseconds.

        Measured from step start to completion.
        Used for performance analysis and optimization.
        """,
        ge=0,
    )

    # === Step Data ===
    step_data: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Step-specific execution data and results.

        Example (Hash Generation):
        {
            "file_hash": "abc123...",
            "file_size_bytes": 1024,
            "performance_grade": "A",
            "hasher_pool_size": 100
        }

        Example (Stamping):
        {
            "stamp_id": "stamp_uuid",
            "stamp_type": "inline",
            "content_length": 500,
            "namespace": "production"
        }

        Example (Intelligence):
        {
            "intelligence_type": "classification",
            "confidence_score": 0.95,
            "response_time_ms": 120
        }
        """,
    )

    # === Error Tracking ===
    error_message: str | None = Field(
        default=None,
        description="""
        Error message if step failed.

        Only populated when status = "FAILED"
        Contains exception details for debugging.
        """,
        max_length=2000,
    )

    # === Temporal Tracking ===
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="""
        Timestamp when this step record was created.

        Automatically set to current UTC time on creation.
        Used for step timeline reconstruction.
        """,
    )

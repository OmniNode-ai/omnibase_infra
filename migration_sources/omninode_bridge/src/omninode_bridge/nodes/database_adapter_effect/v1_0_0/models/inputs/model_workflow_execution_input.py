#!/usr/bin/env python3
"""
ModelWorkflowExecutionInput - Workflow Execution Persistence Input.

Input model for persisting workflow execution records in PostgreSQL.
Supports both INSERT (new workflows) and UPDATE (workflow completion/failure).

ONEX v2.0 Compliance:
- Suffix-based naming: ModelWorkflowExecutionInput
- UUID correlation tracking
- FSM state validation
- Comprehensive field validation with Pydantic v2
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkflowExecutionInput(BaseModel):
    """
    Input model for workflow execution database operations.

    This model represents the data needed to persist workflow execution
    records in the workflow_executions table. It supports both creating
    new workflow records (INSERT) and updating existing records (UPDATE)
    based on correlation_id.

    Database Table: workflow_executions
    Primary Key: id (UUID, auto-generated)
    Unique Key: correlation_id
    Indexes: correlation_id, namespace, current_state

    Workflow States (FSM):
    - PENDING: Workflow created, not yet started
    - PROCESSING: Workflow actively executing
    - COMPLETED: Workflow finished successfully
    - FAILED: Workflow encountered an error

    Event Sources:
    - WORKFLOW_STARTED: Creates new record with PROCESSING state
    - WORKFLOW_COMPLETED: Updates record with COMPLETED state
    - WORKFLOW_FAILED: Updates record with FAILED state

    Example (New Workflow):
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> input_data = ModelWorkflowExecutionInput(
        ...     correlation_id=uuid4(),
        ...     workflow_type="metadata_stamping",
        ...     current_state="PROCESSING",
        ...     namespace="production",
        ...     started_at=datetime.now(timezone.utc),
        ...     metadata={"source": "api", "user_id": "user_123"}
        ... )

    Example (Workflow Completion):
        >>> update_data = ModelWorkflowExecutionInput(
        ...     correlation_id=uuid4(),  # Same as original workflow
        ...     workflow_type="metadata_stamping",
        ...     current_state="COMPLETED",
        ...     namespace="production",
        ...     completed_at=datetime.now(timezone.utc),
        ...     execution_time_ms=1234
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # === Workflow Identity ===
    correlation_id: UUID = Field(
        ...,
        description="""
        UUID correlation identifier for the workflow.

        This serves as the unique identifier for the workflow execution
        and is used as the primary lookup key for UPDATE operations.
        """,
    )

    # === Workflow Classification ===
    workflow_type: str = Field(
        ...,
        description="""
        Type of workflow being executed.

        Common values:
        - "metadata_stamping": Standard stamping workflow
        - "batch_stamping": Batch processing workflow
        - "validation": Stamp validation workflow
        - "intelligence_enrichment": OnexTree intelligence workflow
        """,
        min_length=1,
        max_length=100,
    )

    # === FSM State ===
    current_state: str = Field(
        ...,
        description="""
        Current FSM state of the workflow.

        Valid states: PENDING, PROCESSING, COMPLETED, FAILED

        State transitions:
        - PENDING → PROCESSING: Workflow starts
        - PROCESSING → COMPLETED: Workflow succeeds
        - PROCESSING → FAILED: Workflow encounters error
        """,
        min_length=1,
        max_length=50,
    )

    # === Multi-Tenant Isolation ===
    namespace: str = Field(
        ...,
        description="""
        Namespace for multi-tenant workflow isolation.

        All workflows within the same namespace are grouped together
        for aggregation and analytics purposes.
        """,
        min_length=1,
        max_length=255,
    )

    # === Temporal Tracking ===
    started_at: datetime | None = Field(
        default=None,
        description="""
        Timestamp when the workflow started execution.

        Set when WORKFLOW_STARTED event is processed.
        Null for workflows that haven't started yet (PENDING state).
        """,
    )

    completed_at: datetime | None = Field(
        default=None,
        description="""
        Timestamp when the workflow completed (success or failure).

        Set when WORKFLOW_COMPLETED or WORKFLOW_FAILED event is processed.
        Null for workflows still in progress.
        """,
    )

    # === Performance Metrics ===
    execution_time_ms: int | None = Field(
        default=None,
        description="""
        Total workflow execution time in milliseconds.

        Calculated as: (completed_at - started_at) in milliseconds
        Only available after workflow completion.
        """,
        ge=0,
    )

    # === Error Tracking ===
    error_message: str | None = Field(
        default=None,
        description="""
        Error message if workflow failed.

        Only populated when current_state = "FAILED"
        Contains exception message and stack trace details.
        """,
        max_length=2000,
    )

    # === Extended Metadata ===
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Extended workflow metadata.

        Example:
        {
            "source": "api",
            "user_id": "user_123",
            "api_version": "v1",
            "client_ip": "192.168.1.100",
            "stamping_service_version": "1.0.0",
            "onextree_enabled": true,
            "batch_size": 10
        }
        """,
    )

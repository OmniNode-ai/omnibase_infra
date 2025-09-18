"""Node-specific output model for the workflow coordinator orchestrator."""

from datetime import datetime
from typing import Any
from uuid import UUID

from omnibase_core.model.model_base import ModelBase
from pydantic import Field

from omnibase_infra.models.workflow.model_workflow_coordination_metrics import (
    ModelWorkflowCoordinationMetrics,
)
from omnibase_infra.models.workflow.model_workflow_execution_result import (
    ModelWorkflowExecutionResult,
)
from omnibase_infra.models.workflow.model_workflow_progress_update import (
    ModelWorkflowProgressUpdate,
)


class ModelWorkflowCoordinatorOutput(ModelBase):
    """Output model for workflow coordinator orchestrator operations."""

    success: bool = Field(..., description="Whether the operation succeeded")
    operation_type: str = Field(..., description="Type of operation that was performed")
    correlation_id: UUID = Field(..., description="Correlation ID from the request")
    workflow_id: UUID | None = Field(
        None, description="Workflow ID for the operation",
    )
    execution_result: ModelWorkflowExecutionResult | None = Field(
        None, description="Workflow execution result data",
    )
    progress_update: ModelWorkflowProgressUpdate | None = Field(
        None, description="Current workflow progress information",
    )
    coordination_metrics: ModelWorkflowCoordinationMetrics | None = Field(
        None, description="Current coordination metrics",
    )
    active_workflows: list[dict[str, Any]] = Field(
        default_factory=list, description="List of currently active workflows",
    )
    agent_coordination_status: dict[str, Any] = Field(
        default_factory=dict, description="Status of sub-agent coordination",
    )
    error_message: str | None = Field(
        None, description="Error message if operation failed",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp",
    )

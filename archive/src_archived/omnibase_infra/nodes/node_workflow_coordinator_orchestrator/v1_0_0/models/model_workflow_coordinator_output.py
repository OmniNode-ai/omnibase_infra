"""Node-specific output model for the workflow coordinator orchestrator."""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime
from pydantic import Field

from omnibase_core.model.model_base import ModelBase
from omnibase_infra.models.workflow.model_workflow_execution_result import ModelWorkflowExecutionResult
from omnibase_infra.models.workflow.model_workflow_progress_update import ModelWorkflowProgressUpdate
from omnibase_infra.models.workflow.model_workflow_coordination_metrics import ModelWorkflowCoordinationMetrics


class ModelWorkflowCoordinatorOutput(ModelBase):
    """Output model for workflow coordinator orchestrator operations."""

    success: bool = Field(..., description="Whether the operation succeeded")
    operation_type: str = Field(..., description="Type of operation that was performed")
    correlation_id: UUID = Field(..., description="Correlation ID from the request")
    workflow_id: Optional[UUID] = Field(None, description="Workflow ID for the operation")
    execution_result: Optional[ModelWorkflowExecutionResult] = Field(
        None,
        description="Workflow execution result data"
    )
    progress_update: Optional[ModelWorkflowProgressUpdate] = Field(
        None,
        description="Current workflow progress information"
    )
    coordination_metrics: Optional[ModelWorkflowCoordinationMetrics] = Field(
        None,
        description="Current coordination metrics"
    )
    active_workflows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of currently active workflows"
    )
    agent_coordination_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Status of sub-agent coordination"
    )
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
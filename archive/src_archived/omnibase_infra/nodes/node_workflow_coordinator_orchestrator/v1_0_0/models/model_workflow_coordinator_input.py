"""Node-specific input model for the workflow coordinator orchestrator."""

from typing import Dict, Optional, Any
from uuid import UUID
from pydantic import Field

from omnibase_core.model.model_base import ModelBase
from omnibase_infra.models.workflow.model_workflow_execution_request import ModelWorkflowExecutionRequest


class ModelWorkflowCoordinatorInput(ModelBase):
    """Input model for workflow coordinator orchestrator operations."""

    operation_type: str = Field(
        ...,
        description="Type of workflow coordination operation to perform"
    )
    correlation_id: UUID = Field(..., description="Correlation ID for the operation")
    workflow_request: Optional[ModelWorkflowExecutionRequest] = Field(
        None,
        description="Workflow execution request data"
    )
    workflow_id: Optional[UUID] = Field(None, description="Workflow ID for status operations")
    agent_coordination_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for sub-agent fleet coordination"
    )
    environment: str = Field(
        default="development",
        description="Environment configuration (development, staging, production)"
    )
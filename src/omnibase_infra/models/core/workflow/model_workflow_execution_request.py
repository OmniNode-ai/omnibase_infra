"""Workflow execution request model for ONEX workflow coordination."""

from datetime import datetime
from uuid import UUID

from omnibase_core.enums.enum_environment_type import EnumEnvironmentType
from omnibase_core.enums.enum_priority import EnumPriority
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .model_workflow_execution_context import ModelWorkflowExecutionContext


class ModelWorkflowExecutionRequest(BaseModel):
    """Model for workflow execution requests in the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: UUID = Field(
        ..., description="Unique identifier for the workflow execution",
    )
    correlation_id: UUID = Field(
        ..., description="Correlation ID for tracking across services",
    )
    workflow_type: str = Field(..., description="Type of workflow to execute")
    execution_context: ModelWorkflowExecutionContext = Field(
        default_factory=ModelWorkflowExecutionContext,
        description="Context data for workflow execution",
    )
    agent_coordination_required: bool = Field(
        default=True, description="Whether multi-agent coordination is required",
    )
    priority: EnumPriority = Field(
        default=EnumPriority.NORMAL, description="Execution priority",
    )
    timeout_seconds: int = Field(
        default=300, description="Timeout for workflow execution in seconds",
    )
    retry_count: int = Field(
        default=3, description="Number of retries allowed for failed steps",
    )
    environment: EnumEnvironmentType = Field(default=EnumEnvironmentType.DEVELOPMENT, description="Execution environment")
    background_execution: bool = Field(
        default=False, description="Whether to execute in background",
    )
    progress_tracking_enabled: bool = Field(
        default=True, description="Enable detailed progress tracking",
    )
    sub_agent_fleet_size: int = Field(
        default=1, description="Number of sub-agents to coordinate",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Request creation timestamp",
    )

    @field_validator("execution_context", mode="before")
    @classmethod
    def convert_execution_context(cls, v: dict | ModelWorkflowExecutionContext) -> ModelWorkflowExecutionContext:
        """Convert dict to ModelWorkflowExecutionContext with strict typing."""
        if isinstance(v, dict):
            return ModelWorkflowExecutionContext(**v)
        if isinstance(v, ModelWorkflowExecutionContext):
            return v
        raise ValueError(f"execution_context must be dict or ModelWorkflowExecutionContext, got {type(v)}")

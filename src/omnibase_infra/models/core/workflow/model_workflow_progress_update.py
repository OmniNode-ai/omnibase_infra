"""Workflow progress update model for ONEX workflow coordination."""

from datetime import datetime
from uuid import UUID

from omnibase_core.models.model_base import ModelBase
from pydantic import Field, field_validator, ConfigDict

from .model_workflow_step_details import ModelWorkflowStepDetails
from .model_agent_activity import ModelAgentActivity


class ModelWorkflowProgressUpdate(ModelBase):
    """Model for workflow progress updates from the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: UUID = Field(
        ..., description="Unique identifier for the workflow execution",
    )
    correlation_id: UUID = Field(
        ..., description="Correlation ID for tracking across services",
    )
    current_step: int = Field(..., description="Current step number in the workflow")
    total_steps: int = Field(..., description="Total number of steps in the workflow")
    step_name: str = Field(..., description="Name of the current step being executed")
    step_status: str = Field(
        ..., description="Status of current step (running, completed, failed, waiting)",
    )
    progress_percentage: float = Field(
        ..., description="Overall progress percentage (0-100)",
    )
    elapsed_time_seconds: float = Field(
        ..., description="Elapsed execution time in seconds",
    )
    estimated_remaining_seconds: float | None = Field(
        None, description="Estimated remaining time in seconds",
    )
    step_details: ModelWorkflowStepDetails = Field(
        default_factory=ModelWorkflowStepDetails,
        description="Detailed information about current step",
    )
    agent_activities: list[ModelAgentActivity] = Field(
        default_factory=list, description="Current sub-agent activities",
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="Current performance metrics",
    )
    warning_messages: list[str] = Field(
        default_factory=list, description="Warning messages during execution",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Progress update timestamp",
    )

    @field_validator('step_details', mode='before')
    @classmethod
    def convert_step_details(cls, v: dict | ModelWorkflowStepDetails) -> ModelWorkflowStepDetails:
        """Convert dict to ModelWorkflowStepDetails with strict typing."""
        if isinstance(v, dict):
            return ModelWorkflowStepDetails(**v)
        elif isinstance(v, ModelWorkflowStepDetails):
            return v
        else:
            raise ValueError(f"step_details must be dict or ModelWorkflowStepDetails, got {type(v)}")

    @field_validator('agent_activities', mode='before')
    @classmethod
    def convert_agent_activities(cls, v: list[dict | ModelAgentActivity]) -> list[ModelAgentActivity]:
        """Convert list of dicts to ModelAgentActivity with strict typing."""
        if not isinstance(v, list):
            raise ValueError(f"agent_activities must be a list, got {type(v)}")

        converted_activities = []
        for i, activity in enumerate(v):
            if isinstance(activity, dict):
                converted_activities.append(ModelAgentActivity(**activity))
            elif isinstance(activity, ModelAgentActivity):
                converted_activities.append(activity)
            else:
                raise ValueError(f"agent_activities[{i}] must be dict or ModelAgentActivity, got {type(activity)}")

        return converted_activities

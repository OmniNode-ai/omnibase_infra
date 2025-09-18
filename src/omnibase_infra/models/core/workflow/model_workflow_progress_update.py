"""Workflow progress update model for ONEX workflow coordination."""

from datetime import datetime
from typing import Any, Union
from uuid import UUID

from omnibase_core.model.model_base import ModelBase
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
    step_details: Union[ModelWorkflowStepDetails, dict[str, Any]] = Field(
        default_factory=ModelWorkflowStepDetails,
        description="Detailed information about current step",
    )
    agent_activities: list[Union[ModelAgentActivity, dict[str, Any]]] = Field(
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
    def convert_step_details(cls, v: Any) -> Union[ModelWorkflowStepDetails, dict[str, Any]]:
        """Convert dict to ModelWorkflowStepDetails for backward compatibility."""
        if isinstance(v, dict) and not isinstance(v, ModelWorkflowStepDetails):
            try:
                return ModelWorkflowStepDetails(**v)
            except Exception:
                # If conversion fails, keep as dict for backward compatibility
                return v
        return v

    @field_validator('agent_activities', mode='before')
    @classmethod
    def convert_agent_activities(cls, v: Any) -> list[Union[ModelAgentActivity, dict[str, Any]]]:
        """Convert list of dicts to ModelAgentActivity for backward compatibility."""
        if not isinstance(v, list):
            return v
        
        converted_activities = []
        for activity in v:
            if isinstance(activity, dict) and not isinstance(activity, ModelAgentActivity):
                try:
                    converted_activities.append(ModelAgentActivity(**activity))
                except Exception:
                    # If conversion fails, keep as dict for backward compatibility
                    converted_activities.append(activity)
            else:
                converted_activities.append(activity)
        
        return converted_activities

"""Workflow execution request model for ONEX workflow coordination."""

from datetime import datetime
from typing import Union, Any
from uuid import UUID

from omnibase_core.model.model_base import ModelBase
from pydantic import Field, field_validator, ConfigDict

from .model_workflow_execution_context import ModelWorkflowExecutionContext


class ModelWorkflowExecutionRequest(ModelBase):
    """Model for workflow execution requests in the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: UUID = Field(
        ..., description="Unique identifier for the workflow execution",
    )
    correlation_id: UUID = Field(
        ..., description="Correlation ID for tracking across services",
    )
    workflow_type: str = Field(..., description="Type of workflow to execute")
    execution_context: Union[ModelWorkflowExecutionContext, dict[str, Any]] = Field(
        default_factory=ModelWorkflowExecutionContext,
        description="Context data for workflow execution",
    )
    agent_coordination_required: bool = Field(
        default=True, description="Whether multi-agent coordination is required",
    )
    priority: str = Field(
        default="normal", description="Execution priority (low, normal, high, critical)",
    )
    timeout_seconds: int = Field(
        default=300, description="Timeout for workflow execution in seconds",
    )
    retry_count: int = Field(
        default=3, description="Number of retries allowed for failed steps",
    )
    environment: str = Field(default="development", description="Execution environment")
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

    @field_validator('execution_context', mode='before')
    @classmethod
    def convert_execution_context(cls, v: Any) -> Union[ModelWorkflowExecutionContext, dict[str, Any]]:
        """Convert dict to ModelWorkflowExecutionContext for backward compatibility."""
        if isinstance(v, dict) and not isinstance(v, ModelWorkflowExecutionContext):
            try:
                return ModelWorkflowExecutionContext(**v)
            except Exception:
                # If conversion fails, keep as dict for backward compatibility
                return v
        return v

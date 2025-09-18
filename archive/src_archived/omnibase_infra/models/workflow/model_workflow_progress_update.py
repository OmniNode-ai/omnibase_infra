"""Workflow progress update model for ONEX workflow coordination."""

from datetime import datetime
from typing import Any
from uuid import UUID

from omnibase_core.model.model_base import ModelBase
from pydantic import Field


class ModelWorkflowProgressUpdate(ModelBase):
    """Model for workflow progress updates from the ONEX workflow coordinator."""

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
    step_details: dict[str, Any] = Field(
        default_factory=dict, description="Detailed information about current step",
    )
    agent_activities: list[dict[str, Any]] = Field(
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

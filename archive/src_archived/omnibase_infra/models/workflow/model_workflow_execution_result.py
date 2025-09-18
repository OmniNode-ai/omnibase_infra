"""Workflow execution result model for ONEX workflow coordination."""

from datetime import datetime
from typing import Any
from uuid import UUID

from omnibase_core.model.model_base import ModelBase
from pydantic import Field


class ModelWorkflowExecutionResult(ModelBase):
    """Model for workflow execution results from the ONEX workflow coordinator."""

    workflow_id: UUID = Field(
        ..., description="Unique identifier for the workflow execution",
    )
    correlation_id: UUID = Field(
        ..., description="Correlation ID for tracking across services",
    )
    execution_status: str = Field(
        ...,
        description="Final execution status (completed, failed, timeout, cancelled)",
    )
    success: bool = Field(..., description="Whether the workflow execution succeeded")
    steps_completed: int = Field(
        ..., description="Number of workflow steps completed successfully",
    )
    total_steps: int = Field(..., description="Total number of workflow steps")
    execution_duration_seconds: float = Field(
        ..., description="Total execution duration in seconds",
    )
    result_data: dict[str, Any] = Field(
        default_factory=dict, description="Result data from workflow execution",
    )
    error_details: str | None = Field(
        None, description="Error details if execution failed",
    )
    agent_coordination_summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary of agent coordination activities",
    )
    progress_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Detailed progress tracking history",
    )
    sub_agent_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Results from coordinated sub-agents",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Execution metrics and performance data",
    )
    completed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Execution completion timestamp",
    )

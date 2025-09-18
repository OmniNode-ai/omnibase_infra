"""Workflow execution result model for ONEX workflow coordination."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .model_agent_coordination_summary import ModelAgentCoordinationSummary
from .model_sub_agent_result import ModelSubAgentResult
from .model_workflow_progress_history import ModelWorkflowProgressHistory
from .model_workflow_result_data import ModelWorkflowResultData


class ModelWorkflowExecutionResult(BaseModel):
    """Model for workflow execution results from the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

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
    result_data: ModelWorkflowResultData = Field(
        default_factory=ModelWorkflowResultData,
        description="Result data from workflow execution",
    )
    error_details: str | None = Field(
        None, description="Error details if execution failed",
    )
    agent_coordination_summary: ModelAgentCoordinationSummary = Field(
        default_factory=ModelAgentCoordinationSummary,
        description="Summary of agent coordination activities",
    )
    progress_history: list[ModelWorkflowProgressHistory] = Field(
        default_factory=list, description="Detailed progress tracking history",
    )
    sub_agent_results: list[ModelSubAgentResult] = Field(
        default_factory=list, description="Results from coordinated sub-agents",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Execution metrics and performance data",
    )
    completed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Execution completion timestamp",
    )

    @field_validator("result_data", mode="before")
    @classmethod
    def convert_result_data(cls, v: dict | ModelWorkflowResultData) -> ModelWorkflowResultData:
        """Convert dict to ModelWorkflowResultData with strict typing."""
        if isinstance(v, dict):
            return ModelWorkflowResultData(**v)
        if isinstance(v, ModelWorkflowResultData):
            return v
        raise ValueError(f"result_data must be dict or ModelWorkflowResultData, got {type(v)}")

    @field_validator("agent_coordination_summary", mode="before")
    @classmethod
    def convert_agent_coordination_summary(cls, v: dict | ModelAgentCoordinationSummary) -> ModelAgentCoordinationSummary:
        """Convert dict to ModelAgentCoordinationSummary with strict typing."""
        if isinstance(v, dict):
            return ModelAgentCoordinationSummary(**v)
        if isinstance(v, ModelAgentCoordinationSummary):
            return v
        raise ValueError(f"agent_coordination_summary must be dict or ModelAgentCoordinationSummary, got {type(v)}")

    @field_validator("progress_history", mode="before")
    @classmethod
    def convert_progress_history(cls, v: list[dict | ModelWorkflowProgressHistory]) -> list[ModelWorkflowProgressHistory]:
        """Convert list of dicts to ModelWorkflowProgressHistory with strict typing."""
        if not isinstance(v, list):
            raise ValueError(f"progress_history must be a list, got {type(v)}")

        converted_history = []
        for i, history_entry in enumerate(v):
            if isinstance(history_entry, dict):
                converted_history.append(ModelWorkflowProgressHistory(**history_entry))
            elif isinstance(history_entry, ModelWorkflowProgressHistory):
                converted_history.append(history_entry)
            else:
                raise ValueError(f"progress_history[{i}] must be dict or ModelWorkflowProgressHistory, got {type(history_entry)}")

        return converted_history

    @field_validator("sub_agent_results", mode="before")
    @classmethod
    def convert_sub_agent_results(cls, v: list[dict | ModelSubAgentResult]) -> list[ModelSubAgentResult]:
        """Convert list of dicts to ModelSubAgentResult with strict typing."""
        if not isinstance(v, list):
            raise ValueError(f"sub_agent_results must be a list, got {type(v)}")

        converted_results = []
        for i, agent_result in enumerate(v):
            if isinstance(agent_result, dict):
                converted_results.append(ModelSubAgentResult(**agent_result))
            elif isinstance(agent_result, ModelSubAgentResult):
                converted_results.append(agent_result)
            else:
                raise ValueError(f"sub_agent_results[{i}] must be dict or ModelSubAgentResult, got {type(agent_result)}")

        return converted_results

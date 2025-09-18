"""Workflow execution result model for ONEX workflow coordination."""

from datetime import datetime
from typing import Any, Union
from uuid import UUID

from omnibase_core.model.model_base import ModelBase
from pydantic import Field, field_validator, ConfigDict

from .model_workflow_result_data import ModelWorkflowResultData
from .model_agent_coordination_summary import ModelAgentCoordinationSummary
from .model_workflow_progress_history import ModelWorkflowProgressHistory
from .model_sub_agent_result import ModelSubAgentResult


class ModelWorkflowExecutionResult(ModelBase):
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
    result_data: Union[ModelWorkflowResultData, dict[str, Any]] = Field(
        default_factory=ModelWorkflowResultData,
        description="Result data from workflow execution",
    )
    error_details: str | None = Field(
        None, description="Error details if execution failed",
    )
    agent_coordination_summary: Union[ModelAgentCoordinationSummary, dict[str, Any]] = Field(
        default_factory=ModelAgentCoordinationSummary,
        description="Summary of agent coordination activities",
    )
    progress_history: list[Union[ModelWorkflowProgressHistory, dict[str, Any]]] = Field(
        default_factory=list, description="Detailed progress tracking history",
    )
    sub_agent_results: list[Union[ModelSubAgentResult, dict[str, Any]]] = Field(
        default_factory=list, description="Results from coordinated sub-agents",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Execution metrics and performance data",
    )
    completed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Execution completion timestamp",
    )

    @field_validator('result_data', mode='before')
    @classmethod
    def convert_result_data(cls, v: Any) -> Union[ModelWorkflowResultData, dict[str, Any]]:
        """Convert dict to ModelWorkflowResultData for backward compatibility."""
        if isinstance(v, dict) and not isinstance(v, ModelWorkflowResultData):
            try:
                return ModelWorkflowResultData(**v)
            except Exception:
                # If conversion fails, keep as dict for backward compatibility
                return v
        return v

    @field_validator('agent_coordination_summary', mode='before')
    @classmethod
    def convert_agent_coordination_summary(cls, v: Any) -> Union[ModelAgentCoordinationSummary, dict[str, Any]]:
        """Convert dict to ModelAgentCoordinationSummary for backward compatibility."""
        if isinstance(v, dict) and not isinstance(v, ModelAgentCoordinationSummary):
            try:
                return ModelAgentCoordinationSummary(**v)
            except Exception:
                # If conversion fails, keep as dict for backward compatibility
                return v
        return v

    @field_validator('progress_history', mode='before')
    @classmethod
    def convert_progress_history(cls, v: Any) -> list[Union[ModelWorkflowProgressHistory, dict[str, Any]]]:
        """Convert list of dicts to ModelWorkflowProgressHistory for backward compatibility."""
        if not isinstance(v, list):
            return v
        
        converted_history = []
        for history_entry in v:
            if isinstance(history_entry, dict) and not isinstance(history_entry, ModelWorkflowProgressHistory):
                try:
                    converted_history.append(ModelWorkflowProgressHistory(**history_entry))
                except Exception:
                    # If conversion fails, keep as dict for backward compatibility
                    converted_history.append(history_entry)
            else:
                converted_history.append(history_entry)
        
        return converted_history

    @field_validator('sub_agent_results', mode='before')
    @classmethod
    def convert_sub_agent_results(cls, v: Any) -> list[Union[ModelSubAgentResult, dict[str, Any]]]:
        """Convert list of dicts to ModelSubAgentResult for backward compatibility."""
        if not isinstance(v, list):
            return v
        
        converted_results = []
        for agent_result in v:
            if isinstance(agent_result, dict) and not isinstance(agent_result, ModelSubAgentResult):
                try:
                    converted_results.append(ModelSubAgentResult(**agent_result))
                except Exception:
                    # If conversion fails, keep as dict for backward compatibility
                    converted_results.append(agent_result)
            else:
                converted_results.append(agent_result)
        
        return converted_results

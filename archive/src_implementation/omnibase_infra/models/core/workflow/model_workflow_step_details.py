"""Workflow step details model for ONEX workflow coordination."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enum_agent_type import EnumAgentType
from .enum_workflow_step_category import EnumWorkflowStepCategory
from .enum_workflow_step_priority import EnumWorkflowStepPriority
from .enum_workflow_step_status import EnumWorkflowStepStatus
from .enum_workflow_step_type import EnumWorkflowStepType


class ModelWorkflowStepDetails(BaseModel):
    """Model for detailed workflow step information in the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    # Step identification
    step_id: UUID = Field(
        ...,
        description="Unique identifier for this workflow step",
    )
    step_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name of the workflow step",
    )
    step_type: EnumWorkflowStepType = Field(
        ...,
        description="Type of workflow step",
    )
    step_category: EnumWorkflowStepCategory = Field(
        default=EnumWorkflowStepCategory.PROCESSING,
        description="Category of the workflow step",
    )

    # Step status and progress
    status: EnumWorkflowStepStatus = Field(
        ...,
        description="Current status of the step",
    )
    progress_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Progress percentage for this specific step",
    )

    # Timing information
    started_at: datetime | None = Field(
        None,
        description="Timestamp when step execution started",
    )
    completed_at: datetime | None = Field(
        None,
        description="Timestamp when step execution completed",
    )
    duration_seconds: float | None = Field(
        None,
        ge=0.0,
        description="Step execution duration in seconds",
    )
    estimated_duration_seconds: float | None = Field(
        None,
        ge=0.0,
        description="Estimated duration for step completion",
    )

    # Step configuration
    input_data_size_bytes: int | None = Field(
        None,
        ge=0,
        description="Size of input data in bytes",
    )
    output_data_size_bytes: int | None = Field(
        None,
        ge=0,
        description="Size of output data in bytes",
    )
    memory_usage_mb: float | None = Field(
        None,
        ge=0.0,
        description="Memory usage during step execution in megabytes",
    )
    cpu_usage_percentage: float | None = Field(
        None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage during step execution",
    )

    # Dependencies and relationships
    depends_on_steps: list[UUID] = Field(
        default_factory=list,
        description="List of step IDs that this step depends on",
    )
    blocks_steps: list[UUID] = Field(
        default_factory=list,
        description="List of step IDs that are blocked by this step",
    )

    # Agent information
    assigned_agent_id: UUID | None = Field(
        None,
        description="ID of the agent assigned to execute this step",
    )
    agent_type: EnumAgentType | None = Field(
        None,
        description="Type of agent executing this step",
    )

    # Error handling
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retry attempts made for this step",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries allowed for this step",
    )
    error_message: str | None = Field(
        None,
        description="Error message if step failed",
    )
    error_code: str | None = Field(
        None,
        description="Structured error code for programmatic handling",
    )

    # Output and results
    output_summary: str | None = Field(
        None,
        max_length=1000,
        description="Brief summary of step output or results",
    )
    artifacts_produced: list[str] = Field(
        default_factory=list,
        description="List of artifacts or files produced by this step",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Step-specific metrics and measurements",
    )

    # Metadata and context
    priority: EnumWorkflowStepPriority = Field(
        default=EnumWorkflowStepPriority.NORMAL,
        description="Execution priority for this step",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering steps",
    )
    custom_properties: dict[str, str] = Field(
        default_factory=dict,
        description="Custom properties specific to this step type",
    )

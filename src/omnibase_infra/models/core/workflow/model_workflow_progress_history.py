"""Workflow progress history model for ONEX workflow coordination."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from omnibase_core.models.model_base import ModelBase
from pydantic import Field, ConfigDict


class ModelWorkflowProgressHistory(ModelBase):
    """Model for workflow progress history entries in the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    # History entry identification
    entry_id: UUID = Field(
        ...,
        description="Unique identifier for this progress history entry"
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Sequence number of this entry in the progress history"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when this progress entry was recorded"
    )
    
    # Progress state snapshot
    current_step: int = Field(
        ...,
        ge=0,
        description="Current step number at the time of this entry"
    )
    total_steps: int = Field(
        ...,
        ge=1,
        description="Total number of steps at the time of this entry"
    )
    progress_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Progress percentage at the time of this entry"
    )
    
    # Step information
    step_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the step being executed"
    )
    step_status: str = Field(
        ...,
        description="Status of the step at time of entry",
        examples=["starting", "running", "completed", "failed", "paused", "skipped"]
    )
    step_type: str = Field(
        default="processing",
        description="Type of step being executed",
        examples=["initialization", "processing", "validation", "coordination", "cleanup"]
    )
    
    # Timing information
    elapsed_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed time since workflow start"
    )
    step_elapsed_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time elapsed for the current step"
    )
    estimated_remaining_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated remaining time at time of entry"
    )
    time_since_last_update_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time since last progress update"
    )
    
    # Performance metrics at time of entry
    memory_usage_mb: Optional[float] = Field(
        None,
        ge=0.0,
        description="Memory usage at time of this entry"
    )
    cpu_usage_percentage: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage at time of entry"
    )
    throughput_items_per_second: Optional[float] = Field(
        None,
        ge=0.0,
        description="Processing throughput at time of entry"
    )
    
    # Agent coordination state
    active_agents: int = Field(
        default=0,
        ge=0,
        description="Number of active agents at time of entry"
    )
    idle_agents: int = Field(
        default=0,
        ge=0,
        description="Number of idle agents at time of entry"
    )
    failed_agents: int = Field(
        default=0,
        ge=0,
        description="Number of failed agents at time of entry"
    )
    agent_coordination_efficiency: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Agent coordination efficiency percentage"
    )
    
    # Quality and error tracking
    errors_encountered: int = Field(
        default=0,
        ge=0,
        description="Total number of errors encountered up to this point"
    )
    warnings_generated: int = Field(
        default=0,
        ge=0,
        description="Total number of warnings generated up to this point"
    )
    quality_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Overall quality score at time of entry"
    )
    
    # Data processing statistics
    items_processed: int = Field(
        default=0,
        ge=0,
        description="Total items processed up to this point"
    )
    items_successful: int = Field(
        default=0,
        ge=0,
        description="Items processed successfully up to this point"
    )
    items_failed: int = Field(
        default=0,
        ge=0,
        description="Items that failed processing up to this point"
    )
    processing_rate_per_minute: float = Field(
        default=0.0,
        ge=0.0,
        description="Current processing rate in items per minute"
    )
    
    # Resource utilization snapshot
    network_bytes_sent: int = Field(
        default=0,
        ge=0,
        description="Total network bytes sent up to this point"
    )
    network_bytes_received: int = Field(
        default=0,
        ge=0,
        description="Total network bytes received up to this point"
    )
    disk_io_operations: int = Field(
        default=0,
        ge=0,
        description="Total disk I/O operations up to this point"
    )
    external_api_calls: int = Field(
        default=0,
        ge=0,
        description="Total external API calls made up to this point"
    )
    
    # State changes and events
    state_change_type: str = Field(
        default="progress_update",
        description="Type of state change that triggered this entry",
        examples=["progress_update", "step_completion", "error_recovery", "agent_coordination", "milestone"]
    )
    previous_status: Optional[str] = Field(
        None,
        description="Previous status before this state change"
    )
    event_description: str = Field(
        default="",
        max_length=500,
        description="Description of the event or state change"
    )
    
    # Dependencies and blocking
    waiting_for_dependencies: list[str] = Field(
        default_factory=list,
        description="List of dependencies being waited for"
    )
    blocking_issues: list[str] = Field(
        default_factory=list,
        description="List of issues blocking progress"
    )
    
    # Milestone tracking
    milestone_reached: Optional[str] = Field(
        None,
        description="Milestone reached at this point in execution"
    )
    checkpoint_created: bool = Field(
        default=False,
        description="Whether a checkpoint was created at this point"
    )
    
    # Context and metadata
    execution_phase: str = Field(
        default="execution",
        description="Phase of execution at time of entry",
        examples=["initialization", "execution", "coordination", "finalization", "cleanup"]
    )
    criticality_level: str = Field(
        default="normal",
        description="Criticality level of current operations",
        examples=["low", "normal", "high", "critical"]
    )
    
    # Additional metrics
    custom_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Custom metrics specific to the workflow type"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering history entries"
    )
    
    # Change tracking
    changes_since_last_entry: list[str] = Field(
        default_factory=list,
        description="List of significant changes since the last history entry"
    )
    performance_delta: dict[str, float] = Field(
        default_factory=dict,
        description="Performance changes since last entry"
    )
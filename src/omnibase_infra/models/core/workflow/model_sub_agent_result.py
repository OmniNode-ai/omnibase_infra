"""Sub-agent result model for ONEX workflow coordination."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from omnibase_core.models.model_base import ModelBase
from pydantic import Field, ConfigDict


class ModelSubAgentResult(ModelBase):
    """Model for sub-agent execution results in the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    # Agent identification
    agent_id: UUID = Field(
        ...,
        description="Unique identifier for the sub-agent"
    )
    agent_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name of the sub-agent"
    )
    agent_type: str = Field(
        ...,
        description="Type of sub-agent",
        examples=["specialist", "coordinator", "processor", "validator", "analyzer"]
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Version of the sub-agent software"
    )
    
    # Execution results
    execution_status: str = Field(
        ...,
        description="Final execution status of the sub-agent",
        examples=["completed", "failed", "timeout", "cancelled", "partial_success"]
    )
    success: bool = Field(
        ...,
        description="Whether the sub-agent execution was successful"
    )
    exit_code: int = Field(
        default=0,
        description="Exit code from sub-agent execution (0 = success)"
    )
    completion_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of assigned tasks completed by the sub-agent"
    )
    
    # Timing information
    started_at: datetime = Field(
        ...,
        description="Timestamp when sub-agent execution started"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when sub-agent execution completed"
    )
    execution_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total execution duration in seconds"
    )
    idle_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent idle waiting for tasks or dependencies"
    )
    active_processing_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time actively spent processing tasks"
    )
    
    # Task execution summary
    tasks_assigned: int = Field(
        default=0,
        ge=0,
        description="Total number of tasks assigned to this sub-agent"
    )
    tasks_completed: int = Field(
        default=0,
        ge=0,
        description="Number of tasks completed successfully"
    )
    tasks_failed: int = Field(
        default=0,
        ge=0,
        description="Number of tasks that failed"
    )
    tasks_skipped: int = Field(
        default=0,
        ge=0,
        description="Number of tasks skipped due to dependencies or errors"
    )
    tasks_retried: int = Field(
        default=0,
        ge=0,
        description="Number of tasks that required retry attempts"
    )
    
    # Performance metrics
    average_task_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Average duration per task in seconds"
    )
    throughput_tasks_per_minute: float = Field(
        default=0.0,
        ge=0.0,
        description="Task processing throughput in tasks per minute"
    )
    efficiency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall efficiency score (0-100)"
    )
    
    # Resource utilization
    peak_memory_usage_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Peak memory usage during execution in megabytes"
    )
    average_cpu_usage_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Average CPU usage percentage during execution"
    )
    network_bytes_transferred: int = Field(
        default=0,
        ge=0,
        description="Total network bytes transferred during execution"
    )
    disk_io_operations: int = Field(
        default=0,
        ge=0,
        description="Total disk I/O operations performed"
    )
    
    # Output and artifacts
    output_data_size_bytes: int = Field(
        default=0,
        ge=0,
        description="Size of output data produced in bytes"
    )
    artifacts_created: list[str] = Field(
        default_factory=list,
        description="List of artifacts or files created by the sub-agent"
    )
    output_summary: str = Field(
        default="",
        max_length=1000,
        description="Brief summary of sub-agent output and results"
    )
    
    # Quality metrics
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Quality score of sub-agent output (0-100)"
    )
    validation_results: dict[str, bool] = Field(
        default_factory=dict,
        description="Results of quality validation checks"
    )
    compliance_status: dict[str, bool] = Field(
        default_factory=dict,
        description="Compliance requirement satisfaction status"
    )
    
    # Error handling and debugging
    errors_encountered: int = Field(
        default=0,
        ge=0,
        description="Total number of errors encountered during execution"
    )
    warnings_generated: int = Field(
        default=0,
        ge=0,
        description="Total number of warnings generated during execution"
    )
    critical_errors: int = Field(
        default=0,
        ge=0,
        description="Number of critical errors that required intervention"
    )
    error_details: list[str] = Field(
        default_factory=list,
        description="Detailed error messages and stack traces"
    )
    recovery_actions_taken: int = Field(
        default=0,
        ge=0,
        description="Number of automatic recovery actions taken"
    )
    
    # Dependencies and coordination
    dependencies_resolved: int = Field(
        default=0,
        ge=0,
        description="Number of dependencies successfully resolved"
    )
    coordination_messages_sent: int = Field(
        default=0,
        ge=0,
        description="Number of coordination messages sent to other agents"
    )
    coordination_messages_received: int = Field(
        default=0,
        ge=0,
        description="Number of coordination messages received from other agents"
    )
    blocked_by_dependencies_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time blocked waiting for dependencies"
    )
    
    # Business and domain results
    business_objectives_met: list[str] = Field(
        default_factory=list,
        description="List of business objectives successfully met"
    )
    business_value_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Business value delivered score (0-100)"
    )
    domain_specific_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Domain-specific metrics relevant to the sub-agent's function"
    )
    
    # Learning and adaptation
    patterns_learned: list[str] = Field(
        default_factory=list,
        description="New patterns or insights learned during execution"
    )
    optimization_opportunities: list[str] = Field(
        default_factory=list,
        description="Identified opportunities for performance optimization"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for improving future executions"
    )
    
    # Cost and resource analysis
    estimated_execution_cost: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated cost of sub-agent execution"
    )
    resource_cost_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of costs by resource type"
    )
    cost_efficiency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Cost efficiency score (0-100)"
    )
    
    # Configuration and context
    configuration_used: dict[str, str] = Field(
        default_factory=dict,
        description="Configuration parameters used by the sub-agent"
    )
    environment_context: dict[str, str] = Field(
        default_factory=dict,
        description="Environment context during execution"
    )
    feature_flags_active: dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags that were active during execution"
    )
    
    # Metadata and categorization
    priority_level: str = Field(
        default="normal",
        description="Priority level of sub-agent execution",
        examples=["low", "normal", "high", "critical"]
    )
    execution_mode: str = Field(
        default="standard",
        description="Execution mode used by the sub-agent",
        examples=["standard", "fast", "thorough", "debug", "recovery"]
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering sub-agent results"
    )
    custom_properties: dict[str, str] = Field(
        default_factory=dict,
        description="Custom properties specific to this sub-agent type"
    )
    
    # Relationships and hierarchy
    parent_workflow_id: UUID = Field(
        ...,
        description="ID of the parent workflow this sub-agent was part of"
    )
    parent_agent_id: Optional[UUID] = Field(
        None,
        description="ID of the parent agent that coordinated this sub-agent"
    )
    child_agent_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs of any child agents spawned by this sub-agent"
    )
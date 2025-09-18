"""Agent coordination summary model for ONEX workflow coordination."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from omnibase_core.models.model_base import ModelBase
from pydantic import Field, ConfigDict


class ModelAgentCoordinationSummary(ModelBase):
    """Model for agent coordination summary in the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    # Coordination overview
    coordination_strategy: str = Field(
        ...,
        description="Strategy used for agent coordination",
        examples=["sequential", "parallel", "hybrid", "adaptive", "hierarchical"]
    )
    total_agents_coordinated: int = Field(
        ...,
        ge=0,
        description="Total number of agents coordinated in this workflow"
    )
    active_agents_peak: int = Field(
        default=0,
        ge=0,
        description="Peak number of agents active simultaneously"
    )
    coordination_complexity: str = Field(
        default="simple",
        description="Complexity level of coordination required",
        examples=["simple", "moderate", "complex", "highly_complex"]
    )
    
    # Execution timing
    coordination_start_time: datetime = Field(
        ...,
        description="Timestamp when agent coordination began"
    )
    coordination_end_time: Optional[datetime] = Field(
        None,
        description="Timestamp when agent coordination completed"
    )
    total_coordination_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time spent on coordination activities"
    )
    average_agent_response_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Average response time from agents"
    )
    
    # Agent performance metrics
    successful_agent_executions: int = Field(
        default=0,
        ge=0,
        description="Number of successful agent task executions"
    )
    failed_agent_executions: int = Field(
        default=0,
        ge=0,
        description="Number of failed agent task executions"
    )
    timeout_agent_executions: int = Field(
        default=0,
        ge=0,
        description="Number of agent executions that timed out"
    )
    retry_attempts_total: int = Field(
        default=0,
        ge=0,
        description="Total number of retry attempts across all agents"
    )
    
    # Coordination efficiency
    coordination_efficiency_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall coordination efficiency percentage"
    )
    parallel_execution_time_saved_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time saved through parallel execution"
    )
    resource_utilization_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Average resource utilization across agents"
    )
    idle_time_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of time agents were idle"
    )
    
    # Communication and messaging
    total_messages_exchanged: int = Field(
        default=0,
        ge=0,
        description="Total number of messages exchanged between agents"
    )
    coordination_messages: int = Field(
        default=0,
        ge=0,
        description="Number of coordination-specific messages"
    )
    heartbeat_messages: int = Field(
        default=0,
        ge=0,
        description="Number of heartbeat messages exchanged"
    )
    error_messages: int = Field(
        default=0,
        ge=0,
        description="Number of error messages reported"
    )
    average_message_size_bytes: float = Field(
        default=0.0,
        ge=0.0,
        description="Average size of coordination messages in bytes"
    )
    
    # Resource management
    peak_memory_usage_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Peak memory usage across all coordinated agents"
    )
    peak_cpu_usage_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Peak CPU usage across all coordinated agents"
    )
    network_bandwidth_used_mbps: float = Field(
        default=0.0,
        ge=0.0,
        description="Network bandwidth used for coordination"
    )
    disk_io_operations: int = Field(
        default=0,
        ge=0,
        description="Total disk I/O operations during coordination"
    )
    
    # Error handling and recovery
    coordination_errors: int = Field(
        default=0,
        ge=0,
        description="Number of coordination-level errors encountered"
    )
    agent_recovery_actions: int = Field(
        default=0,
        ge=0,
        description="Number of agent recovery actions performed"
    )
    deadlock_incidents: int = Field(
        default=0,
        ge=0,
        description="Number of deadlock incidents detected and resolved"
    )
    coordination_restarts: int = Field(
        default=0,
        ge=0,
        description="Number of times coordination had to be restarted"
    )
    
    # Agent-specific summaries
    agent_performance_summary: dict[str, float] = Field(
        default_factory=dict,
        description="Performance scores by agent ID (0-100 scale)"
    )
    agent_utilization_summary: dict[str, float] = Field(
        default_factory=dict,
        description="Utilization percentages by agent ID"
    )
    agent_error_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Error counts by agent ID"
    )
    agent_task_completion_times: dict[str, float] = Field(
        default_factory=dict,
        description="Average task completion times by agent ID (seconds)"
    )
    
    # Dependency management
    dependency_resolution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent resolving agent dependencies"
    )
    dependency_conflicts: int = Field(
        default=0,
        ge=0,
        description="Number of dependency conflicts encountered"
    )
    circular_dependencies_detected: int = Field(
        default=0,
        ge=0,
        description="Number of circular dependencies detected"
    )
    
    # Quality and compliance
    coordination_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall coordination quality score (0-100)"
    )
    sla_compliance_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Service level agreement compliance percentage"
    )
    governance_violations: int = Field(
        default=0,
        ge=0,
        description="Number of governance policy violations"
    )
    
    # Optimization insights
    bottleneck_agents: list[str] = Field(
        default_factory=list,
        description="Agent IDs that were identified as bottlenecks"
    )
    optimization_opportunities: list[str] = Field(
        default_factory=list,
        description="Identified opportunities for coordination optimization"
    )
    recommended_improvements: list[str] = Field(
        default_factory=list,
        description="Recommended improvements for future coordinations"
    )
    
    # Cost analysis
    coordination_cost_estimate: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated cost of coordination activities"
    )
    resource_cost_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by resource type"
    )
    cost_efficiency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Cost efficiency score (0-100)"
    )
    
    # Metadata and context
    coordination_version: str = Field(
        default="1.0.0",
        description="Version of coordination framework used"
    )
    environment: str = Field(
        default="production",
        description="Environment where coordination took place"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing coordination summaries"
    )
    custom_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Custom coordination metrics specific to workflow type"
    )
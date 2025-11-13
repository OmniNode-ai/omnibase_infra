"""
Data models for agent context distribution.

This module provides type-safe models for packaging and distributing
agent-specific context in parallel coordination workflows.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class CoordinationMetadata(BaseModel):
    """
    Coordination session metadata for agent identification.

    Attributes:
        session_id: Unique coordination session identifier
        coordination_id: Coordination workflow identifier
        agent_id: Unique agent identifier
        agent_role: Agent's role in workflow (e.g., "model_generator", "validator")
        parent_workflow_id: Optional parent workflow ID for nested workflows
        created_at: Context creation timestamp
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(description="Coordination session ID")
    coordination_id: str = Field(description="Coordination workflow ID")
    agent_id: str = Field(description="Unique agent identifier")
    agent_role: str = Field(description="Agent role in workflow")
    parent_workflow_id: Optional[str] = Field(
        default=None, description="Parent workflow ID (for nested workflows)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Context creation timestamp",
    )


class SharedIntelligence(BaseModel):
    """
    Shared intelligence distributed to all agents.

    Contains common data structures, patterns, and registries that
    all agents need access to for consistency.

    Attributes:
        type_registry: Type name to definition mapping
        pattern_library: Available patterns by category
        validation_rules: Shared validation rules
        naming_conventions: Naming conventions for consistency
        dependency_graph: Dependency relationships between components
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_registry: dict[str, Any] = Field(
        default_factory=dict, description="Type name → definition"
    )
    pattern_library: dict[str, list[str]] = Field(
        default_factory=dict, description="Pattern category → pattern list"
    )
    validation_rules: dict[str, Any] = Field(
        default_factory=dict, description="Shared validation rules"
    )
    naming_conventions: dict[str, str] = Field(
        default_factory=dict, description="Naming conventions"
    )
    dependency_graph: dict[str, list[str]] = Field(
        default_factory=dict, description="Component dependencies"
    )


class AgentAssignment(BaseModel):
    """
    Agent's specific assignment in workflow.

    Attributes:
        objective: High-level objective for agent
        tasks: List of specific tasks to complete
        input_data: Input data for agent processing
        dependencies: Agent IDs this agent depends on
        output_requirements: Expected output format/requirements
        success_criteria: Criteria for successful completion
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    objective: str = Field(description="High-level objective")
    tasks: list[str] = Field(description="Specific tasks to complete")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for processing"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Dependent agent IDs"
    )
    output_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Expected output requirements"
    )
    success_criteria: dict[str, Any] = Field(
        default_factory=dict, description="Success criteria"
    )


class CoordinationProtocols(BaseModel):
    """
    Coordination protocols for agent communication.

    Attributes:
        update_interval_ms: Frequency of status updates in milliseconds
        heartbeat_interval_ms: Heartbeat interval in milliseconds
        status_update_channel: Channel for status updates (e.g., "state", "kafka")
        result_delivery_channel: Channel for result delivery
        error_reporting_channel: Channel for error reporting
        coordination_endpoint: Optional coordination service endpoint
    """

    model_config = ConfigDict(frozen=True)

    update_interval_ms: int = Field(
        default=5000, description="Status update interval (ms)", ge=100
    )
    heartbeat_interval_ms: int = Field(
        default=10000, description="Heartbeat interval (ms)", ge=100
    )
    status_update_channel: str = Field(
        default="state", description="Status update channel"
    )
    result_delivery_channel: str = Field(
        default="state", description="Result delivery channel"
    )
    error_reporting_channel: str = Field(
        default="state", description="Error reporting channel"
    )
    coordination_endpoint: Optional[str] = Field(
        default=None, description="Coordination service endpoint"
    )


class ResourceAllocation(BaseModel):
    """
    Resource allocation limits for agent.

    Attributes:
        max_execution_time_ms: Maximum execution time in milliseconds
        max_retry_attempts: Maximum number of retry attempts
        max_memory_mb: Maximum memory allocation in MB
        quality_threshold: Minimum quality threshold (0.0-1.0)
        timeout_ms: Operation timeout in milliseconds
        concurrency_limit: Maximum concurrent operations
    """

    model_config = ConfigDict(frozen=True)

    max_execution_time_ms: int = Field(
        default=300000, description="Max execution time (ms)", ge=1000
    )
    max_retry_attempts: int = Field(
        default=3, description="Max retry attempts", ge=0
    )
    max_memory_mb: int = Field(default=512, description="Max memory (MB)", ge=1)
    quality_threshold: float = Field(
        default=0.8, description="Quality threshold", ge=0.0, le=1.0
    )
    timeout_ms: int = Field(default=30000, description="Operation timeout (ms)", ge=100)
    concurrency_limit: int = Field(
        default=10, description="Max concurrent operations", ge=1
    )


class AgentContext(BaseModel):
    """
    Complete context package for an agent.

    This is the primary model distributed to each agent in a parallel
    coordination workflow. It contains all information the agent needs
    to execute its assigned tasks.

    Attributes:
        coordination_metadata: Session and agent identification
        shared_intelligence: Common data structures and patterns
        agent_assignment: Agent's specific assignment
        coordination_protocols: Communication and update protocols
        resource_allocation: Resource limits and constraints
        context_version: Context version for tracking updates
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    coordination_metadata: CoordinationMetadata = Field(
        description="Coordination metadata"
    )
    shared_intelligence: SharedIntelligence = Field(description="Shared intelligence")
    agent_assignment: AgentAssignment = Field(description="Agent assignment")
    coordination_protocols: CoordinationProtocols = Field(
        description="Coordination protocols"
    )
    resource_allocation: ResourceAllocation = Field(description="Resource allocation")
    context_version: int = Field(default=1, description="Context version", ge=1)


class ContextDistributionMetrics(BaseModel):
    """
    Metrics for context distribution tracking.

    Attributes:
        distribution_time_ms: Time to distribute context to agent
        context_size_bytes: Size of distributed context in bytes
        agent_count: Number of agents receiving context
        shared_intelligence_size_bytes: Size of shared intelligence
        success: Whether distribution was successful
    """

    model_config = ConfigDict(frozen=False)

    distribution_time_ms: float = Field(description="Distribution time (ms)", ge=0.0)
    context_size_bytes: int = Field(description="Context size (bytes)", ge=0)
    agent_count: int = Field(description="Number of agents", ge=1)
    shared_intelligence_size_bytes: int = Field(
        description="Shared intelligence size (bytes)", ge=0
    )
    success: bool = Field(description="Distribution success")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


class ContextUpdateRequest(BaseModel):
    """
    Request to update shared intelligence across agents.

    Attributes:
        coordination_id: Coordination workflow ID
        update_type: Type of update (e.g., "type_registry", "pattern_library")
        update_data: Updated data
        target_agents: Optional list of agent IDs to update (None = all)
        increment_version: Whether to increment context version
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    coordination_id: str = Field(description="Coordination workflow ID")
    update_type: str = Field(description="Update type")
    update_data: dict[str, Any] = Field(description="Updated data")
    target_agents: Optional[list[str]] = Field(
        default=None, description="Target agent IDs (None = all)"
    )
    increment_version: bool = Field(
        default=True, description="Increment context version"
    )

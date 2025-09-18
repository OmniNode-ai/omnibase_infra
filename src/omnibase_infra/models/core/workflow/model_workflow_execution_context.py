"""Workflow execution context model for ONEX workflow coordination."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkflowExecutionContext(BaseModel):
    """Model for workflow execution context data in the ONEX workflow coordinator."""

    model_config = ConfigDict(extra="forbid")

    # Core execution parameters
    environment: str = Field(
        default="development",
        description="Execution environment (development, staging, production)",
        examples=["development", "staging", "production"],
    )
    user_id: UUID | None = Field(
        None,
        description="ID of the user who initiated the workflow",
    )
    session_id: UUID | None = Field(
        None,
        description="Session identifier for tracking user interactions",
    )

    # Workflow configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries allowed for failed operations",
    )
    retry_delay_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=300.0,
        description="Delay between retry attempts in seconds",
    )
    checkpoint_interval_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Interval for creating execution checkpoints",
    )

    # Agent coordination settings
    agent_coordination_strategy: str = Field(
        default="sequential",
        description="Strategy for coordinating multiple agents",
        examples=["sequential", "parallel", "hybrid", "adaptive"],
    )
    max_concurrent_agents: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of agents that can run concurrently",
    )
    agent_timeout_seconds: int = Field(
        default=180,
        ge=30,
        le=1800,
        description="Timeout for individual agent operations",
    )

    # Performance and resource settings
    memory_limit_mb: int | None = Field(
        None,
        ge=256,
        le=32768,
        description="Memory limit for workflow execution in megabytes",
    )
    cpu_limit_cores: float | None = Field(
        None,
        ge=0.1,
        le=16.0,
        description="CPU limit for workflow execution in cores",
    )
    disk_limit_mb: int | None = Field(
        None,
        ge=100,
        le=102400,
        description="Disk space limit for workflow execution in megabytes",
    )

    # Monitoring and observability
    monitoring_enabled: bool = Field(
        default=True,
        description="Whether to enable detailed monitoring and telemetry",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for workflow execution",
        examples=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable distributed tracing",
    )

    # Security and compliance
    security_context: dict[str, str] = Field(
        default_factory=dict,
        description="Security-related context (permissions, tokens, etc.)",
    )
    compliance_requirements: list[str] = Field(
        default_factory=list,
        description="List of compliance requirements to enforce",
        examples=["GDPR", "SOX", "HIPAA", "PCI-DSS"],
    )

    # Custom workflow parameters
    workflow_parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Workflow-specific parameters as string key-value pairs",
    )
    feature_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags for enabling/disabling functionality",
    )

    # Metadata
    created_by: str = Field(
        default="system",
        description="Entity that created this execution context",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when context was created",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering workflows",
    )

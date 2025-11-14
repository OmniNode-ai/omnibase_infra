"""Workflow coordination metrics model for ONEX workflow coordination."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelWorkflowCoordinationMetrics(BaseModel):
    """Model for workflow coordination metrics from the ONEX workflow coordinator."""

    coordinator_id: str = Field(
        ..., description="Identifier for the workflow coordinator instance",
    )
    active_workflows: int = Field(
        default=0, description="Number of currently active workflows",
    )
    completed_workflows_today: int = Field(
        default=0, description="Number of workflows completed today",
    )
    failed_workflows_today: int = Field(
        default=0, description="Number of workflows failed today",
    )
    average_execution_time_seconds: float = Field(
        default=0.0, description="Average workflow execution time",
    )
    agent_coordination_success_rate: float = Field(
        default=1.0, description="Success rate of agent coordination (0-1)",
    )
    sub_agent_fleet_utilization: float = Field(
        default=0.0, description="Utilization rate of sub-agent fleet (0-1)",
    )
    background_tasks_queue_size: int = Field(
        default=0, description="Number of background tasks in queue",
    )
    progress_tracking_active: bool = Field(
        default=True, description="Whether progress tracking is active",
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="Detailed performance metrics",
    )
    resource_utilization: dict[str, float] = Field(
        default_factory=dict, description="Resource utilization metrics",
    )
    error_statistics: dict[str, int] = Field(
        default_factory=dict, description="Error occurrence statistics",
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Metrics last updated timestamp",
    )

    @field_validator("agent_coordination_success_rate", "sub_agent_fleet_utilization")
    @classmethod
    def validate_rate_range(cls, v: float) -> float:
        """Ensure rate values are between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Rate values must be between 0.0 and 1.0")
        return v

    @field_validator("completed_workflows_today", "failed_workflows_today", "active_workflows")
    @classmethod
    def validate_non_negative_counts(cls, v: int) -> int:
        """Ensure count values are non-negative."""
        if v < 0:
            raise ValueError("Count values must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_metric_relationships(self) -> "ModelWorkflowCoordinationMetrics":
        """Validate relationships between metrics."""
        # Total workflows processed today should be reasonable
        total_today = self.completed_workflows_today + self.failed_workflows_today
        if total_today > 10000:  # Reasonable daily limit
            raise ValueError("Total workflows today seems unreasonably high")

        # Success rate validation
        if total_today > 0:
            actual_success_rate = self.completed_workflows_today / total_today
            # Allow some tolerance in coordination success rate vs actual success rate
            if abs(self.agent_coordination_success_rate - actual_success_rate) > 0.5:
                # Log warning but don't fail - coordination success might be different from workflow success
                pass

        # Fleet utilization should be reasonable
        if self.sub_agent_fleet_utilization > 0.95:
            # Very high utilization - might indicate overload
            pass

        return self

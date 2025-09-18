"""Workflow coordination metrics model for ONEX workflow coordination."""

from datetime import datetime

from omnibase_core.model.model_base import ModelBase
from pydantic import Field


class ModelWorkflowCoordinationMetrics(ModelBase):
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

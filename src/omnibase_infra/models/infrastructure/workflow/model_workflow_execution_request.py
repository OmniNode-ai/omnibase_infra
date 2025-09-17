"""Workflow execution request model for ONEX workflow coordination."""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from omnibase_core.model.model_base import ModelBase


class ModelWorkflowExecutionRequest(ModelBase):
    """Model for workflow execution requests in the ONEX workflow coordinator."""

    workflow_id: UUID = Field(..., description="Unique identifier for the workflow execution")
    correlation_id: UUID = Field(..., description="Correlation ID for tracking across services")
    workflow_type: str = Field(..., description="Type of workflow to execute")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Context data for workflow execution")
    agent_coordination_required: bool = Field(default=True, description="Whether multi-agent coordination is required")
    priority: str = Field(default="normal", description="Execution priority (low, normal, high, critical)")
    timeout_seconds: int = Field(default=300, description="Timeout for workflow execution in seconds")
    retry_count: int = Field(default=3, description="Number of retries allowed for failed steps")
    environment: str = Field(default="development", description="Execution environment")
    background_execution: bool = Field(default=False, description="Whether to execute in background")
    progress_tracking_enabled: bool = Field(default=True, description="Enable detailed progress tracking")
    sub_agent_fleet_size: int = Field(default=1, description="Number of sub-agents to coordinate")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Request creation timestamp")
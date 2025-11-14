"""Output model for NodeOmniInfraOrchestrator."""

from uuid import UUID

from pydantic import BaseModel, Field


class ModelOmniInfraOrchestratorOutput(BaseModel):
    """Output from infrastructure orchestrator - workflow execution result."""

    workflow_executed: bool = Field(..., description="Whether workflow was executed")
    workflow_name: str = Field(..., description="Name of executed workflow")
    workflow_result: dict = Field(default_factory=dict, description="Result from workflow")
    correlation_id: UUID = Field(..., description="Request correlation ID")
    timestamp: float = Field(..., description="Response timestamp")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "workflow_executed": True,
                "workflow_name": "health_check_workflow",
                "workflow_result": {
                    "overall_health_status": "healthy",
                    "adapters_checked": ["postgres", "kafka", "consul"]
                },
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": 1699999999.999,
            }
        }

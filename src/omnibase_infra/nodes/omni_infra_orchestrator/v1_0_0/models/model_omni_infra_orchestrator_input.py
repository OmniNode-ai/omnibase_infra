"""Input model for NodeOmniInfraOrchestrator."""

from uuid import UUID

from pydantic import BaseModel, Field


class ModelOmniInfraOrchestratorInput(BaseModel):
    """Input for infrastructure orchestrator - intent or workflow trigger."""

    operation_type: str = Field(
        ...,
        description="Type of operation: process_intent | trigger_workflow | query_workflow_status"
    )
    intent_type: str | None = Field(
        None,
        description="Intent from reducer (if operation_type=process_intent)"
    )
    workflow_name: str | None = Field(
        None,
        description="Workflow to trigger (if operation_type=trigger_workflow)"
    )
    workflow_params: dict = Field(
        default_factory=dict,
        description="Parameters for workflow execution"
    )
    correlation_id: UUID = Field(..., description="Request correlation ID")
    timestamp: float = Field(..., description="Request timestamp")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "operation_type": "process_intent",
                "intent_type": "infrastructure_health_degraded",
                "workflow_params": {},
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": 1699999999.999,
            }
        }

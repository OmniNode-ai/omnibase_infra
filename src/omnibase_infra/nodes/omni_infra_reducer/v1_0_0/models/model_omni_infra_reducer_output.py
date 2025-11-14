"""Output model for NodeOmniInfraReducer."""

from uuid import UUID

from pydantic import BaseModel, Field


class ModelOmniInfraReducerOutput(BaseModel):
    """Output from infrastructure reducer - state update result."""

    state_updated: bool = Field(..., description="Whether state was stored in DB")
    state_id: UUID = Field(..., description="ID of stored state record")
    intents_emitted: list[str] = Field(default_factory=list, description="Intents emitted to orchestrator")
    correlation_id: UUID = Field(..., description="Request correlation ID")
    timestamp: float = Field(..., description="Response timestamp")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "state_updated": True,
                "state_id": "123e4567-e89b-12d3-a456-426614174000",
                "intents_emitted": ["infrastructure_health_degraded"],
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": 1699999999.999,
            }
        }

"""Input model for NodeOmniInfraReducer."""

from uuid import UUID

from pydantic import BaseModel, Field


class ModelOmniInfraReducerInput(BaseModel):
    """Input for infrastructure reducer - event from adapter."""

    event_type: str = Field(..., description="Type of infrastructure event")
    adapter_source: str = Field(..., description="Source adapter: postgres | kafka | consul")
    event_payload: dict = Field(..., description="Event-specific payload")
    correlation_id: UUID = Field(..., description="Request correlation ID")
    timestamp: float = Field(..., description="Event timestamp")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "event_type": "POSTGRES_HEALTH_RESPONSE",
                "adapter_source": "postgres",
                "event_payload": {
                    "health_status": "healthy",
                    "pool_size": 10,
                    "active_connections": 3,
                },
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": 1699999999.999,
            }
        }

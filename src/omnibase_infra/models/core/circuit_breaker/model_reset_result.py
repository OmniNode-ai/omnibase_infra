"""Reset Result Model."""

from datetime import datetime
from uuid import UUID

from omnibase_core.enums.intelligence.enum_circuit_breaker_state import (
    EnumCircuitBreakerState,
)
from pydantic import BaseModel, Field


class ModelResetResult(BaseModel):
    """Result for reset circuit operations."""

    reset_successful: bool = Field(
        description="Whether the reset was successful",
    )

    previous_state: EnumCircuitBreakerState = Field(
        description="Circuit breaker state before reset",
    )

    new_state: EnumCircuitBreakerState = Field(
        description="Circuit breaker state after reset",
    )

    metrics_reset: bool = Field(
        description="Whether metrics were also reset",
    )

    events_cleared_from_queue: int = Field(
        ge=0,
        description="Number of events cleared from queue",
    )

    dead_letter_queue_cleared: bool = Field(
        description="Whether dead letter queue was cleared",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
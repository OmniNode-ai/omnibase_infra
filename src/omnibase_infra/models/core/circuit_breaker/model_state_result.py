"""State Result Model."""

from datetime import datetime
from uuid import UUID

from omnibase_core.enums.intelligence.enum_circuit_breaker_state import (
    EnumCircuitBreakerState,
)
from pydantic import BaseModel, Field


class ModelStateResult(BaseModel):
    """Result for get state operations."""

    current_state: EnumCircuitBreakerState = Field(
        description="Current circuit breaker state",
    )

    failure_count: int = Field(
        ge=0,
        description="Current failure count",
    )

    success_count: int = Field(
        ge=0,
        description="Current success count",
    )

    last_failure_time: datetime | None = Field(
        default=None,
        description="Timestamp of last failure",
    )

    time_in_current_state_seconds: float = Field(
        ge=0.0,
        description="How long in current state (seconds)",
    )

    next_state_transition_estimate: datetime | None = Field(
        default=None,
        description="Estimated time of next state transition",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
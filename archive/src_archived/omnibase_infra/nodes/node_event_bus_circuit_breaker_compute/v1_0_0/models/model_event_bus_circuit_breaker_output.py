"""Event Bus Circuit Breaker Output Model.

Node-specific output model for the circuit breaker compute node.
"""

from datetime import datetime
from uuid import UUID

from omnibase_core.enums.intelligence.enum_circuit_breaker_state import (
    EnumCircuitBreakerState,
)
from pydantic import BaseModel, Field

from omnibase_infra.models.circuit_breaker.model_circuit_breaker_metrics import (
    ModelCircuitBreakerMetrics,
)
from omnibase_infra.models.circuit_breaker.model_circuit_breaker_result import (
    ModelHealthStatusResult,
    ModelPublishEventResult,
    ModelResetResult,
    ModelStateResult,
)


class ModelEventBusCircuitBreakerOutput(BaseModel):
    """Output model for event bus circuit breaker operations."""

    success: bool = Field(
        description="Whether the operation succeeded",
    )

    operation_type: str = Field(
        description="Type of operation that was performed",
    )

    correlation_id: UUID = Field(
        description="Correlation ID from the request",
    )

    result: (
        ModelPublishEventResult
        | ModelStateResult
        | ModelResetResult
        | ModelHealthStatusResult
        | None
    ) = Field(
        default=None,
        description="Operation-specific result data",
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )

    circuit_breaker_state: EnumCircuitBreakerState = Field(
        description="Current circuit breaker state",
    )

    metrics: ModelCircuitBreakerMetrics | None = Field(
        default=None,
        description="Current circuit breaker metrics",
    )

    timestamp: datetime = Field(
        description="Response timestamp",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

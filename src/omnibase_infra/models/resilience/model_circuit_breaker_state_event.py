# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event model for circuit breaker state transitions.

Emitted whenever a circuit breaker transitions between states:
  CLOSED → OPEN, OPEN → HALF_OPEN, or HALF_OPEN → CLOSED.

Producer: CircuitBreakerEventPublisher
Consumer: omnidash /circuit-breaker dashboard (OMN-5293)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_circuit_state import EnumCircuitState


class ModelCircuitBreakerStateEvent(BaseModel):
    """Payload for a circuit breaker state transition event.

    Attributes:
        service_name: Identifier of the service whose circuit changed.
        state: New circuit breaker state after the transition.
        previous_state: State before the transition.
        failure_count: Failure counter at the time of the transition.
        threshold: Configured failure threshold for this circuit.
        timestamp: UTC time the transition occurred.
        correlation_id: Optional trace correlation ID.
    """

    service_name: str = Field(description="Service identifier for the circuit breaker")
    state: EnumCircuitState = Field(description="New circuit breaker state")
    previous_state: EnumCircuitState = Field(
        description="Previous circuit breaker state before transition"
    )
    failure_count: int = Field(ge=0, description="Failure count at time of transition")
    threshold: int = Field(ge=1, description="Configured failure threshold")
    timestamp: datetime = Field(description="UTC timestamp of the state transition")
    correlation_id: UUID | None = Field(
        default=None, description="Optional correlation ID for distributed tracing"
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )


__all__: list[str] = ["ModelCircuitBreakerStateEvent"]

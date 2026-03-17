# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Circuit breaker state-transition event publisher.

Publishes ModelCircuitBreakerStateEvent to the Kafka topic
``onex.evt.omnibase-infra.circuit-breaker.v1`` whenever a circuit breaker
transitions between states (CLOSED ↔ OPEN ↔ HALF_OPEN).

Design:
    - Stateless: caller passes all transition context on each call.
    - Fault-tolerant: publish errors are logged but not re-raised so that
      a bus outage never blocks the circuit breaker's own state machine.
    - Dependency-injected: accepts any ProtocolEventBusLike implementation
      for testability (pass an in-memory bus in unit tests).

Usage::

    publisher = CircuitBreakerEventPublisher(event_bus=kafka_bus)

    # Inside _record_circuit_failure / _reset_circuit_breaker overrides:
    await publisher.publish_transition(
        service_name="kafka.production",
        new_state=EnumCircuitState.OPEN,
        previous_state=EnumCircuitState.CLOSED,
        failure_count=5,
        threshold=5,
    )

Related:
    - SUFFIX_CIRCUIT_BREAKER_STATE: topic constant (OMN-5293)
    - ModelCircuitBreakerStateEvent: event payload model
    - omnidash /circuit-breaker: consumer dashboard
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_circuit_state import EnumCircuitState
from omnibase_infra.models.resilience.model_circuit_breaker_state_event import (
    ModelCircuitBreakerStateEvent,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.topics import SUFFIX_CIRCUIT_BREAKER_STATE

logger = logging.getLogger(__name__)


class CircuitBreakerEventPublisher:
    """Publishes circuit breaker state transition events to the event bus.

    All publish errors are caught and logged; they never propagate to the
    circuit breaker itself so that a bus outage cannot block state transitions.

    Attributes:
        _event_bus: The underlying event bus implementation.
        _topic: Resolved Kafka topic string for circuit breaker events.
    """

    def __init__(self, event_bus: ProtocolEventBusLike) -> None:
        self._event_bus = event_bus
        self._topic: str = SUFFIX_CIRCUIT_BREAKER_STATE

    async def publish_transition(
        self,
        service_name: str,
        new_state: EnumCircuitState,
        previous_state: EnumCircuitState,
        failure_count: int,
        threshold: int,
        correlation_id: UUID | None = None,
    ) -> None:
        """Publish a state-transition event for a circuit breaker.

        Args:
            service_name: Service identifier (e.g. "kafka.production").
            new_state: The circuit breaker's new state after the transition.
            previous_state: The state before the transition.
            failure_count: Current failure count at time of transition.
            threshold: Configured failure threshold for this circuit.
            correlation_id: Optional trace correlation ID.
        """
        now = datetime.now(UTC)
        event = ModelCircuitBreakerStateEvent(
            service_name=service_name,
            state=new_state,
            previous_state=previous_state,
            failure_count=failure_count,
            threshold=threshold,
            timestamp=now,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=json.loads(event.model_dump_json()),
            envelope_timestamp=now,
            correlation_id=correlation_id if correlation_id else uuid4(),
            source="circuit_breaker_event_publisher",
        )

        try:
            await self._event_bus.publish_envelope(
                envelope,  # type: ignore[arg-type]
                topic=self._topic,
            )
            logger.debug(
                "Circuit breaker state published",
                extra={
                    "service_name": service_name,
                    "new_state": new_state.value,
                    "previous_state": previous_state.value,
                    "failure_count": failure_count,
                    "topic": self._topic,
                },
            )
        except Exception as exc:  # noqa: BLE001 — never block circuit breaker on bus errors
            logger.error(  # noqa: TRY400 — avoid leaking stack traces in error logs
                "Failed to publish circuit breaker state event: %s",
                exc,
                extra={
                    "service_name": service_name,
                    "new_state": new_state.value,
                    "topic": self._topic,
                },
            )


__all__: list[str] = ["CircuitBreakerEventPublisher"]

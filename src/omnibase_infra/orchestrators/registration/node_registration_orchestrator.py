# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Node.

This module provides the NodeRegistrationOrchestrator, the FIRST orchestrator
node in omnibase_infra. The orchestrator processes registration workflow events
and emits decision events based on projection state queries.

Architectural Constraints:
    - Returns EVENTS ONLY (no intents, no projections)
    - Performs NO I/O (projection reads are read-only database lookups)
    - Uses `now` parameter for all time decisions (never datetime.now())
    - Uses projection reader for all state queries

Event Flow:
    Input Events:
        - ModelNodeIntrospectionEvent: Node announces itself
        - ModelRuntimeTick: Periodic timeout detection trigger
        - ModelNodeRegistrationAcked: Node acknowledges registration (command)

    Output Events:
        - ModelNodeRegistrationInitiated: Registration attempt started
        - ModelNodeRegistrationAckTimedOut: Ack deadline passed
        - ModelNodeLivenessExpired: Liveness deadline passed
        - ModelNodeRegistrationAckReceived: Ack processed successfully
        - ModelNodeBecameActive: Node transitioned to active state

Thread Safety:
    This class is NOT thread-safe. Each instance should be used by a single
    consumer thread. For concurrent processing, create multiple instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.orchestrators.registration.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.orchestrators.registration.handlers.handler_node_registration_acked import (
    HandlerNodeRegistrationAcked,
)
from omnibase_infra.orchestrators.registration.handlers.handler_runtime_tick import (
    HandlerRuntimeTick,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _resolve_correlation_id(
    explicit: UUID | None,
    envelope: ModelEventEnvelope[object],
) -> UUID:
    """Resolve correlation ID from explicit param, envelope, or generate new.

    Resolution order:
    1. Explicit correlation_id parameter if provided
    2. Envelope's correlation_id attribute if present
    3. Generate new UUID4 as fallback

    Args:
        explicit: Explicitly passed correlation ID (highest priority).
        envelope: Event envelope that may contain correlation_id.

    Returns:
        Resolved correlation ID for tracing.
    """
    return explicit or getattr(envelope, "correlation_id", None) or uuid4()


class NodeRegistrationOrchestrator:
    """Registration orchestrator - emits EVENTS only, no I/O.

    The Registration Orchestrator is the first orchestrator node in omnibase_infra.
    It receives registration workflow events and makes decisions based on
    the current projection state.

    Handler Routing:
        Events are routed to specialized handlers based on their type:
        - ModelNodeIntrospectionEvent -> HandlerNodeIntrospected
        - ModelRuntimeTick -> HandlerRuntimeTick
        - ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked

    Projection Queries:
        All state queries are performed via ProjectionReaderRegistration,
        which provides read-only access to the registration projection table.
        The orchestrator never scans Kafka topics directly.

    Time Injection:
        All handlers receive the `now` parameter for time-based decisions.
        This ensures deterministic behavior and supports testing.

    Example:
        >>> from asyncpg import create_pool
        >>> pool = await create_pool(dsn)
        >>> reader = ProjectionReaderRegistration(pool)
        >>> orchestrator = NodeRegistrationOrchestrator(reader)
        >>>
        >>> # Handle an introspection event
        >>> events = await orchestrator.handle(
        ...     envelope=introspection_envelope,
        ...     now=datetime.now(UTC),
        ...     correlation_id=uuid4(),
        ... )
        >>> # events contains ModelNodeRegistrationInitiated if new registration

    Attributes:
        _projection_reader: Reader for registration projection state.
        _handlers: Mapping of event types to handler instances.
    """

    def __init__(self, projection_reader: ProjectionReaderRegistration) -> None:
        """Initialize the orchestrator with a projection reader.

        Args:
            projection_reader: Reader for registration projection state.
                              Used by handlers to query current entity state.
        """
        self._projection_reader = projection_reader

        # Initialize handlers
        self._handler_introspected = HandlerNodeIntrospected(projection_reader)
        self._handler_runtime_tick = HandlerRuntimeTick(projection_reader)
        self._handler_registration_acked = HandlerNodeRegistrationAcked(
            projection_reader
        )

    @property
    def projection_reader(self) -> ProjectionReaderRegistration:
        """Get the projection reader (for testing/introspection)."""
        return self._projection_reader

    async def handle(
        self,
        envelope: ModelEventEnvelope[object],
        now: datetime,
        correlation_id: UUID | None = None,
    ) -> list[BaseModel]:
        """Route to appropriate handler and return emitted events.

        This is the main entry point for the orchestrator. It extracts the
        payload from the envelope, routes to the appropriate handler based
        on type, and returns the list of events to emit.

        Args:
            envelope: The event envelope containing the payload to process.
                     The payload type determines which handler is invoked.
            now: Injected current time for all timeout/deadline decisions.
                 Handlers MUST use this instead of calling datetime.now().
            correlation_id: Optional correlation ID for distributed tracing.
                           If not provided, uses envelope's correlation_id or
                           generates a new one.

        Returns:
            List of event models to emit. May be empty if no action needed.
            All events are Pydantic BaseModel instances from the registration
            domain events module.

        Raises:
            ValueError: If the envelope payload type is not supported.
            RuntimeHostError: If projection queries fail (propagated from reader).

        Example:
            >>> events = await orchestrator.handle(
            ...     envelope=tick_envelope,
            ...     now=tick.now,
            ...     correlation_id=tick.correlation_id,
            ... )
            >>> for event in events:
            ...     await event_bus.publish(event)
        """
        # Resolve correlation ID
        corr_id = _resolve_correlation_id(correlation_id, envelope)

        # Extract payload
        payload = envelope.payload

        # Route based on payload type
        if isinstance(payload, ModelNodeIntrospectionEvent):
            logger.debug(
                "Routing to HandlerNodeIntrospected",
                extra={"node_id": str(payload.node_id), "correlation_id": str(corr_id)},
            )
            return await self._handler_introspected.handle(
                event=payload,
                now=now,
                correlation_id=corr_id,
            )

        if isinstance(payload, ModelRuntimeTick):
            logger.debug(
                "Routing to HandlerRuntimeTick",
                extra={
                    "tick_id": str(payload.tick_id),
                    "correlation_id": str(corr_id),
                },
            )
            return await self._handler_runtime_tick.handle(
                tick=payload,
                now=now,
                correlation_id=corr_id,
            )

        if isinstance(payload, ModelNodeRegistrationAcked):
            logger.debug(
                "Routing to HandlerNodeRegistrationAcked",
                extra={
                    "node_id": str(payload.node_id),
                    "correlation_id": str(corr_id),
                },
            )
            return await self._handler_registration_acked.handle(
                command=payload,
                now=now,
                correlation_id=corr_id,
            )

        # Unknown payload type
        payload_type = type(payload).__name__
        raise ValueError(
            f"Unsupported payload type for registration orchestrator: {payload_type}"
        )

    async def handle_introspection(
        self,
        event: ModelNodeIntrospectionEvent,
        now: datetime,
        correlation_id: UUID | None = None,
    ) -> list[BaseModel]:
        """Direct handler for introspection events (convenience method).

        This method provides direct access to the introspection handler
        without requiring envelope wrapping. Useful for testing and
        integration scenarios.

        Args:
            event: The introspection event to process.
            now: Injected current time.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            List of events to emit (e.g., ModelNodeRegistrationInitiated).
        """
        corr_id = correlation_id or event.correlation_id
        return await self._handler_introspected.handle(
            event=event,
            now=now,
            correlation_id=corr_id,
        )

    async def handle_runtime_tick(
        self,
        tick: ModelRuntimeTick,
        now: datetime | None = None,
        correlation_id: UUID | None = None,
    ) -> list[BaseModel]:
        """Direct handler for runtime ticks (convenience method).

        This method provides direct access to the runtime tick handler
        without requiring envelope wrapping. Useful for testing and
        integration scenarios.

        Args:
            tick: The runtime tick to process.
            now: Injected current time. If not provided, uses tick.now.
            correlation_id: Optional correlation ID for tracing.
                           If not provided, uses tick.correlation_id.

        Returns:
            List of timeout events (ack timeout or liveness expired).
        """
        effective_now = now if now is not None else tick.now
        corr_id = correlation_id or tick.correlation_id
        return await self._handler_runtime_tick.handle(
            tick=tick,
            now=effective_now,
            correlation_id=corr_id,
        )

    async def handle_registration_ack(
        self,
        command: ModelNodeRegistrationAcked,
        now: datetime,
        correlation_id: UUID | None = None,
    ) -> list[BaseModel]:
        """Direct handler for registration ack commands (convenience method).

        This method provides direct access to the registration ack handler
        without requiring envelope wrapping. Useful for testing and
        integration scenarios.

        Args:
            command: The registration ack command to process.
            now: Injected current time.
            correlation_id: Optional correlation ID for tracing.
                           If not provided, uses command.correlation_id.

        Returns:
            List of events (ack received, became active) or empty if invalid state.
        """
        corr_id = correlation_id or command.correlation_id
        return await self._handler_registration_acked.handle(
            command=command,
            now=now,
            correlation_id=corr_id,
        )


__all__: list[str] = ["NodeRegistrationOrchestrator"]

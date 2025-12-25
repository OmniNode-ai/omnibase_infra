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
    - Uses container injection for dependency resolution

Container Injection:
    The orchestrator uses ModelONEXContainer for dependency injection. The
    container must have ProjectionReaderRegistration registered before
    instantiation. Use wire_infrastructure_services() or manual registration.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> # Register projection reader
        >>> await container.service_registry.register_instance(
        ...     interface=ProjectionReaderRegistration,
        ...     instance=projection_reader,
        ...     scope="global",
        ... )
        >>> # Create orchestrator
        >>> orchestrator = NodeRegistrationOrchestrator(container)

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

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

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


class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Registration orchestrator - extends NodeOrchestrator with container injection.

    The Registration Orchestrator is the first orchestrator node in omnibase_infra.
    It receives registration workflow events and makes decisions based on
    the current projection state. Extends NodeOrchestrator for workflow-driven
    coordination with contract.yaml configuration.

    Container Injection:
        This orchestrator uses ModelONEXContainer for dependency injection.
        The container should have ProjectionReaderRegistration registered.
        If not registered, use set_projection_reader() after instantiation.

    Handler Routing:
        Events are routed to specialized handlers based on their type:
        - ModelNodeIntrospectionEvent -> HandlerNodeIntrospected
        - ModelRuntimeTick -> HandlerRuntimeTick
        - ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked

        The base NodeOrchestrator class also provides workflow_coordination
        support via contract.yaml for declarative handler routing.

    Projection Queries:
        All state queries are performed via ProjectionReaderRegistration,
        which provides read-only access to the registration projection table.
        The orchestrator never scans Kafka topics directly.

    Time Injection:
        All handlers receive the `now` parameter for time-based decisions.
        This ensures deterministic behavior and supports testing.

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> # Register projection reader in container
        >>> await container.service_registry.register_instance(
        ...     interface=ProjectionReaderRegistration,
        ...     instance=projection_reader,
        ...     scope="global",
        ... )
        >>> orchestrator = NodeRegistrationOrchestrator(container)
        >>>
        >>> # Handle an introspection event
        >>> events = await orchestrator.handle(
        ...     envelope=introspection_envelope,
        ...     now=datetime.now(UTC),
        ...     correlation_id=uuid4(),
        ... )
        >>> # events contains ModelNodeRegistrationInitiated if new registration

    Attributes:
        _projection_reader: Reader for registration projection state (resolved from container).
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the orchestrator with container injection.

        Extends NodeOrchestrator and resolves ProjectionReaderRegistration
        from the container's service registry. If the projection reader is
        not registered in the container, it can be set later using
        set_projection_reader().

        Args:
            container: ONEX dependency injection container. Should have
                      ProjectionReaderRegistration registered, but this
                      is not strictly required at construction time.
        """
        super().__init__(container)

        # Resolve projection reader from container (optional at init time)
        # If not available, caller must use set_projection_reader()
        self._projection_reader: ProjectionReaderRegistration | None = (
            self._resolve_projection_reader(container)
        )

        # Initialize handlers lazily when projection reader is available
        self._handler_introspected: HandlerNodeIntrospected | None = None
        self._handler_runtime_tick: HandlerRuntimeTick | None = None
        self._handler_registration_acked: HandlerNodeRegistrationAcked | None = None

        # Wire handlers if projection reader was resolved
        if self._projection_reader is not None:
            self._wire_handlers(self._projection_reader)

    def _resolve_projection_reader(
        self, container: ModelONEXContainer
    ) -> ProjectionReaderRegistration | None:
        """Resolve ProjectionReaderRegistration from container.

        Uses the container's service registry to resolve the projection reader.
        Supports both mock containers (for testing) and real containers.

        Args:
            container: ONEX container with service registry.

        Returns:
            ProjectionReaderRegistration instance from container, or None if
            not registered. Caller should use set_projection_reader() if None.
        """
        try:
            # Support mock containers with direct _projection_reader attribute
            if hasattr(container, "_projection_reader"):
                reader = container._projection_reader
                if isinstance(reader, ProjectionReaderRegistration):
                    return reader
                return None

            # Try get_service_optional from container (returns None if not found)
            if hasattr(container, "get_service_optional"):
                result = container.get_service_optional(
                    ProjectionReaderRegistration,
                    service_name="projection_reader.registration",
                )
                if isinstance(result, ProjectionReaderRegistration):
                    return result

            # Support mock containers with resolve method that returns sync
            if hasattr(container, "service_registry"):
                registry = container.service_registry
                if registry is not None and hasattr(registry, "resolve"):
                    # Mock containers may have sync resolve
                    result = registry.resolve(ProjectionReaderRegistration)
                    if isinstance(result, ProjectionReaderRegistration):
                        return result

            # Not found - caller should use set_projection_reader()
            return None
        except Exception:
            # Resolution failed - caller should use set_projection_reader()
            return None

    def _wire_handlers(self, projection_reader: ProjectionReaderRegistration) -> None:
        """Wire handlers with the projection reader.

        Args:
            projection_reader: Reader for registration projection state.
        """
        self._handler_introspected = HandlerNodeIntrospected(projection_reader)
        self._handler_runtime_tick = HandlerRuntimeTick(projection_reader)
        self._handler_registration_acked = HandlerNodeRegistrationAcked(
            projection_reader
        )

    def set_projection_reader(self, reader: ProjectionReaderRegistration) -> None:
        """Set the projection reader for state queries.

        Use this method when the projection reader is not registered in
        the container at construction time. This is required before calling
        handle() or any handler methods.

        Args:
            reader: ProjectionReaderRegistration instance.

        Example:
            >>> pool = await asyncpg.create_pool(dsn)
            >>> reader = ProjectionReaderRegistration(pool)
            >>> orchestrator.set_projection_reader(reader)
        """
        self._projection_reader = reader
        self._wire_handlers(reader)

    @property
    def has_projection_reader(self) -> bool:
        """Check if projection reader is configured."""
        return self._projection_reader is not None

    @property
    def projection_reader(self) -> ProjectionReaderRegistration | None:
        """Get the projection reader (for testing/introspection)."""
        return self._projection_reader

    def _ensure_handlers_configured(self) -> None:
        """Ensure handlers are configured before use.

        Raises:
            RuntimeError: If projection reader is not configured.
        """
        if self._projection_reader is None or self._handler_introspected is None:
            raise RuntimeError(
                "Projection reader not configured. "
                "Call set_projection_reader() before handling events."
            )

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
            RuntimeError: If projection reader is not configured.
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
        # Ensure handlers are configured
        self._ensure_handlers_configured()

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
            assert self._handler_introspected is not None  # Checked above
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
            assert self._handler_runtime_tick is not None  # Checked above
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
            assert self._handler_registration_acked is not None  # Checked above
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

        Raises:
            RuntimeError: If projection reader is not configured.
        """
        self._ensure_handlers_configured()
        assert self._handler_introspected is not None  # Checked above
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

        Raises:
            RuntimeError: If projection reader is not configured.
        """
        self._ensure_handlers_configured()
        assert self._handler_runtime_tick is not None  # Checked above
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

        Raises:
            RuntimeError: If projection reader is not configured.
        """
        self._ensure_handlers_configured()
        assert self._handler_registration_acked is not None  # Checked above
        corr_id = correlation_id or command.correlation_id
        return await self._handler_registration_acked.handle(
            command=command,
            now=now,
            correlation_id=corr_id,
        )


__all__: list[str] = ["NodeRegistrationOrchestrator"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for NodeIntrospectionEvent - canonical registration trigger.

This handler processes NodeIntrospectionEvent payloads from nodes announcing
their presence in the cluster. It queries the projection for current state
and delegates the registration decision to RegistrationReducerService.

Architecture:
    This handler follows the Reducer-Authoritative pattern:

    1. Handler reads projection state (direct I/O via ProjectionReaderRegistration)
    2. Handler delegates decision to RegistrationReducerService (pure function)
    3. Handler returns ModelHandlerOutput with events and intents from the decision

    The reducer service owns all state decision logic, intent construction,
    and event creation. The handler is a thin coordination layer that bridges
    the I/O boundary (projection read) with the pure decision function.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different event instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
    - OMN-944 (F1): Registration Projection Schema
    - OMN-892: 2-Way Registration E2E Integration Test
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from uuid import UUID, uuid4

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
    ServiceIntrospectionTopicStore,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.utils import validate_timezone_aware_with_context

logger = logging.getLogger(__name__)


class HandlerNodeIntrospected:
    """Handler for NodeIntrospectionEvent - canonical registration trigger.

    This handler processes introspection events from nodes announcing
    themselves to the cluster. It queries the current projection state
    and delegates the registration decision to RegistrationReducerService.

    Dependency Injection:
        This handler receives its dependencies (``projection_reader`` and
        ``reducer``) via constructor injection. The dependencies are wired
        through the registry's ``handler_dependencies`` pattern in
        ``RegistryInfraNodeRegistrationOrchestrator.create_registry()``.

        This is an intentional design choice: handlers accept concrete
        dependencies directly rather than resolving them from
        ``ModelONEXContainer`` at runtime. This provides:

        - **Type safety**: Constructor signatures are validated by mypy
        - **Testability**: Tests inject mocks directly without container setup
        - **Explicitness**: Each handler's dependencies are visible in its signature

        See ``registry_infra_node_registration_orchestrator.py`` module docstring
        section "Handler Dependency Map - Design Trade-off" for the full rationale.

    Reducer-Authoritative Pattern:
        All state decision logic, intent construction, and event creation
        are encapsulated in RegistrationReducerService.decide_introspection().
        This handler is responsible only for:

        1. Reading projection state (I/O)
        2. Calling the reducer service (pure function)
        3. Returning the decision as ModelHandlerOutput

    State Decision Matrix (owned by RegistrationReducerService):
        | Current State       | Action                          |
        |---------------------|----------------------------------|
        | None (new node)     | Emit registration events         |
        | LIVENESS_EXPIRED    | Emit registration events         |
        | REJECTED            | Emit registration events         |
        | ACK_TIMED_OUT       | Emit registration events         |
        | PENDING_REGISTRATION| No-op (already processing)       |
        | ACCEPTED            | No-op (waiting for ack)          |
        | AWAITING_ACK        | No-op (waiting for ack)          |
        | ACK_RECEIVED        | No-op (transitioning to active)  |
        | ACTIVE              | No-op (use heartbeat instead)    |

    Attributes:
        _projection_reader: Reader for registration projection state.
        _reducer: Pure-function reducer service for registration decisions.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> reducer = RegistrationReducerService(ack_timeout_seconds=30.0)
        >>> handler = HandlerNodeIntrospected(projection_reader, reducer)
        >>> output = await handler.handle(envelope)
        >>> # output.intents contains ModelIntent objects for effect layer
        >>> # output.events contains registration events from reducer decision
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        reducer: RegistrationReducerService,
        topic_store: ServiceIntrospectionTopicStore | None = None,
    ) -> None:
        """Initialize the handler with a projection reader and reducer service.

        Dependencies are injected via the registry's ``handler_dependencies``
        pattern in ``RegistryInfraNodeRegistrationOrchestrator.create_registry()``.
        The registry resolves dependencies from ``ModelONEXContainer`` and passes
        them as explicit constructor arguments. This allows tests to inject mocks
        directly without wiring a full DI container while maintaining container-
        managed lifecycle in production.

        Args:
            projection_reader: Reader for querying registration projection state.
            reducer: Pure-function reducer service that encapsulates all
                registration decision logic (state checks, event creation,
                intent construction). Configuration such as ack_timeout_seconds
                and consul_enabled lives on the reducer, not on this handler.
            topic_store: Optional shared in-memory store for accumulating
                event_bus publish topics from introspection events. When provided,
                this handler populates the store on every introspection event so
                that HandlerCatalogRequest can assemble catalog responses.
                When None, topic accumulation is skipped (backward compatible).
        """
        self._projection_reader = projection_reader
        self._reducer = reducer
        self._topic_store = topic_store

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-node-introspected"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelNodeIntrospectionEvent"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler.

        Returns NODE_HANDLER because this handler processes node-level
        introspection events (not infrastructure plumbing).
        """
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns EFFECT because this handler reads from the PostgreSQL
        projection store via ProjectionReaderRegistration.get_entity_state().
        Writes are intent-based (OMN-2050) but reads are direct I/O.
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelNodeIntrospectionEvent],
    ) -> ModelHandlerOutput[object]:
        """Process introspection event and decide on registration.

        Queries the current projection state for the node and delegates
        the registration decision to RegistrationReducerService.

        Returns ModelHandlerOutput with:
        - events: Registration events from reducer decision (if initiating)
        - intents: Effect layer intents from reducer decision (if initiating)

        Args:
            envelope: Event envelope containing ModelNodeIntrospectionEvent payload.

        Returns:
            ModelHandlerOutput with events and intents for effect layer execution.

        Raises:
            RuntimeHostError: If projection query fails (propagated).
            ProtocolConfigurationError: If envelope timestamp is naive (no timezone info).
        """
        start_time = time.perf_counter()

        # Extract from envelope
        event = envelope.payload
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Validate timezone-awareness for time injection pattern
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="handle_introspection_event",
            target_name="handler.node_introspected",
            correlation_id=correlation_id,
        )
        validate_timezone_aware_with_context(now, ctx)

        node_id = event.node_id

        # Populate introspection topic store for catalog responder (OMN-2923)
        # Done unconditionally (before decision) so the catalog always reflects
        # the latest known event_bus config regardless of registration state.
        if self._topic_store is not None and event.event_bus is not None:
            publish_topics = event.event_bus.publish_topic_strings
            await self._topic_store.update_node(str(node_id), publish_topics)
        elif self._topic_store is not None:
            # Node sent introspection without event_bus config; record empty set
            # so count_nodes_missing_event_bus() works correctly.
            await self._topic_store.update_node(str(node_id), [])

        # Query current projection state
        projection = await self._projection_reader.get_entity_state(
            entity_id=node_id,
            domain="registration",
            correlation_id=correlation_id,
        )

        # Delegate decision to reducer service (pure function, no I/O)
        decision = self._reducer.decide_introspection(
            projection=projection,
            event=event,
            correlation_id=correlation_id,
            now=now,
        )

        if decision.action == "no_op":
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelHandlerOutput(
                input_envelope_id=envelope.envelope_id,
                correlation_id=correlation_id,
                handler_id=self.handler_id,
                node_kind=self.node_kind,
                events=(),
                intents=(),
                projections=(),
                result=None,
                processing_time_ms=processing_time_ms,
                timestamp=now,
            )

        logger.info(
            "Emitting registration events with %d intents",
            len(decision.intents),
            extra={
                "node_id": str(node_id),
                "intent_types": [i.intent_type for i in decision.intents],
                "correlation_id": str(correlation_id),
            },
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=decision.events,
            intents=decision.intents,
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )


__all__: list[str] = ["HandlerNodeIntrospected"]

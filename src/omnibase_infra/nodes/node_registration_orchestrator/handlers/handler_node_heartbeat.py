# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Heartbeat Handler for Registration Orchestrator.

Processes NodeHeartbeatReceived events and emits intents to update the
registration projection with `last_heartbeat_at` and extended `liveness_deadline`.

This handler delegates all decision-making to RegistrationReducerService
(Paradigm B: intent-based). The reducer returns a ModelReducerDecision
containing intents that the effect layer executes.

This handler is part of the 2-way registration pattern where nodes periodically
send heartbeats to maintain their ACTIVE registration state.

Related Tickets:
    - OMN-1006: Add last_heartbeat_at for liveness expired event reporting
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-881: Node introspection with configurable topics
    - OMN-1102: Refactor to ProtocolMessageHandler signature
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_context import (
    ModelReducerContext,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.utils import validate_timezone_aware_with_context

if TYPE_CHECKING:
    from omnibase_infra.projectors import ProjectionReaderRegistration

logger = logging.getLogger(__name__)


class HandlerNodeHeartbeat:
    """Handler for processing node heartbeat events.

    Processes ModelNodeHeartbeatEvent events and delegates the heartbeat
    decision to RegistrationReducerService.decide_heartbeat(). The reducer
    returns a ModelReducerDecision containing:
    - UPDATE intent for PostgreSQL (last_heartbeat_at, liveness_deadline)
    - No events (heartbeats don't produce domain events)

    The handler returns these intents via ModelHandlerOutput for the effect
    layer to execute.

    ONEX Contract Compliance:
        This handler belongs to an ORCHESTRATOR node, so it returns result=None
        per ONEX contract rules (ORCHESTRATOR nodes use events[] and intents[]
        only, not result).

    Error Handling:
        - If no projection exists, reducer returns no_op and handler logs warning
        - Only ACTIVE nodes should receive heartbeats; other states log warnings
        - No direct I/O is performed; all writes are via intents

    Coroutine Safety:
        This handler is stateless and coroutine-safe. The projection reader
        is assumed to be coroutine-safe (it uses connection pools).

    Example:
        >>> from omnibase_infra.projectors import ProjectionReaderRegistration
        >>> from omnibase_infra.nodes.node_registration_orchestrator.services import (
        ...     RegistrationReducerService,
        ... )
        >>> reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        >>> handler = HandlerNodeHeartbeat(
        ...     projection_reader=reader,
        ...     reducer=reducer,
        ... )
        >>> output = await handler.handle(envelope)
        >>> # Intents contain the UPDATE payload for the effect layer
        >>> for intent in output.intents:
        ...     print(f"Intent: {intent.intent_type} -> {intent.target}")
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        reducer: RegistrationReducerService,
    ) -> None:
        """Initialize the heartbeat handler.

        Args:
            projection_reader: Projection reader for looking up node state.
            reducer: RegistrationReducerService for pure-function heartbeat
                decisions. The reducer's liveness_window_seconds configuration
                controls how long to extend liveness_deadline from the heartbeat
                timestamp.
        """
        self._projection_reader = projection_reader
        self._reducer = reducer

    @property
    def handler_id(self) -> str:
        """Return unique identifier for this handler."""
        return "handler-node-heartbeat"

    @property
    def category(self) -> EnumMessageCategory:
        """Return the message category this handler processes."""
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Return the set of message types this handler processes."""
        return {"ModelNodeHeartbeatEvent"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Return the node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler.

        Returns NODE_HANDLER because this handler processes node-level
        heartbeat events (not infrastructure plumbing).
        """
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns EFFECT because this handler reads from the PostgreSQL
        projection store via ProjectionReaderRegistration.get_entity_state().
        Writes are intent-based but reads are direct I/O.
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelNodeHeartbeatEvent],
    ) -> ModelHandlerOutput[object]:
        """Process a node heartbeat event.

        Looks up the registration projection by node_id and delegates the
        heartbeat decision to RegistrationReducerService.decide_heartbeat().
        Returns the reducer's intents via ModelHandlerOutput for the effect
        layer to execute.

        ONEX Contract Compliance:
            This handler belongs to an ORCHESTRATOR node, so it returns
            result=None per ONEX contract rules. The intents tuple contains
            the UPDATE payload for PostgreSQL.

        Args:
            envelope: Event envelope containing the heartbeat event payload.

        Returns:
            ModelHandlerOutput with result=None and intents from the reducer.
            The intents contain ModelPayloadPostgresUpdateRegistration payloads
            for the effect layer to execute.
        """
        start_time = time.perf_counter()

        # Extract from envelope
        event = envelope.payload
        now = envelope.envelope_timestamp
        correlation_id = envelope.correlation_id or uuid4()
        domain = "registration"

        # Validate timezone-awareness for time injection pattern
        error_ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="handle_heartbeat_event",
            target_name="handler.node_heartbeat",
            correlation_id=correlation_id,
        )
        validate_timezone_aware_with_context(now, error_ctx)

        # Look up current projection
        projection = await self._projection_reader.get_entity_state(
            entity_id=event.node_id,
            domain=domain,
            correlation_id=correlation_id,
        )

        if projection is None:
            logger.warning(
                "Heartbeat received for unknown node",
                extra={
                    "node_id": str(event.node_id),
                    "correlation_id": str(correlation_id),
                },
            )
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

        # Check if node is in a state that should receive heartbeats
        if not projection.current_state.is_active():
            logger.warning(
                "Heartbeat received for non-active node",
                extra={
                    "node_id": str(event.node_id),
                    "current_state": projection.current_state.value,
                    "correlation_id": str(correlation_id),
                },
            )
            # Still process the heartbeat to update tracking, but log the warning
            # This can happen during state transitions or race conditions

        # Decision: Should we update heartbeat?
        ctx = ModelReducerContext(correlation_id=correlation_id, now=now)
        decision = self._reducer.decide_heartbeat(
            projection=projection,
            node_id=event.node_id,
            heartbeat_timestamp=event.timestamp,
            ctx=ctx,
        )

        if decision.action == "no_op":
            logger.warning(
                "Heartbeat decision: no_op",
                extra={
                    "node_id": str(event.node_id),
                    "reason": decision.reason,
                    "correlation_id": str(correlation_id),
                },
            )
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

        logger.debug(
            "Heartbeat processed, emitting update intent",
            extra={
                "node_id": str(event.node_id),
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


__all__ = [
    "HandlerNodeHeartbeat",
]

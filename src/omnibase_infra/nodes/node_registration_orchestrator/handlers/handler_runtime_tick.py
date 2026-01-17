# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for RuntimeTick - timeout detection.

This handler processes RuntimeTick events from the runtime scheduler
and detects overdue ack and liveness deadlines. It queries the projection
for entities that need timeout events emitted.

Detection Logic:
    For Ack Timeout:
        - Query projection for entities with overdue ack deadlines
        - Use projection.needs_ack_timeout_event() for deduplication
        - Emit NodeRegistrationAckTimedOut for each overdue entity

    For Liveness Expiry:
        - Query projection for entities with overdue liveness deadlines
        - Use projection.needs_liveness_timeout_event() for deduplication
        - Emit NodeLivenessExpired for each overdue entity

Deduplication (per C2 Durable Timeout Handling):
    The projection stores emission markers (ack_timeout_emitted_at,
    liveness_timeout_emitted_at) to prevent duplicate timeout events.
    The projection reader filters out already-emitted timeouts.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different tick instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-940 (F0): Projector Execution Model
    - OMN-1102: Refactor to ProtocolMessageHandler signature
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.models.registration.events.model_node_liveness_expired import (
    ModelNodeLivenessExpired,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_timed_out import (
    ModelNodeRegistrationAckTimedOut,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.utils import validate_timezone_aware_with_context

logger = logging.getLogger(__name__)


class HandlerRuntimeTick:
    """Handler for RuntimeTick - timeout detection.

    This handler processes runtime tick events and scans the projection
    for entities with overdue deadlines. It emits timeout events for
    entities that need them, using projection emission markers for
    deduplication.

    Timeout Detection:
        The handler performs two scans on each tick:
        1. Ack timeout: Find entities waiting for ack past their deadline
        2. Liveness expiry: Find active entities past their liveness deadline

    Projection Queries:
        Uses dedicated projection reader methods that filter by:
        - Deadline < now (deadline has passed)
        - Emission marker IS NULL (not yet emitted)
        - Appropriate state (AWAITING_ACK for ack, ACTIVE for liveness)

    Attributes:
        _projection_reader: Reader for registration projection state.

    Example:
        >>> from datetime import datetime, timezone
        >>> from uuid import uuid4
        >>> from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
        >>> # Use explicit timestamps (time injection pattern) - not datetime.now()
        >>> tick_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        >>> runtime_tick = ModelRuntimeTick(
        ...     now=tick_time,
        ...     tick_id=uuid4(),
        ...     sequence_number=1,
        ...     scheduled_at=tick_time,
        ...     correlation_id=uuid4(),
        ...     scheduler_id="runtime-001",
        ...     tick_interval_ms=1000,
        ... )
        >>> handler = HandlerRuntimeTick(projection_reader)
        >>> events = await handler.handle(
        ...     tick=runtime_tick,
        ...     now=tick_time,
        ...     correlation_id=runtime_tick.correlation_id,
        ... )
        >>> # Output events use injected `now` for emitted_at:
        >>> # ModelNodeRegistrationAckTimedOut(emitted_at=tick_time, ...)
        >>> # ModelNodeLivenessExpired(emitted_at=tick_time, last_heartbeat_at=<datetime|None>, ...)
        >>> # Note: last_heartbeat_at is None if no heartbeats were ever received
    """

    def __init__(self, projection_reader: ProjectionReaderRegistration) -> None:
        """Initialize the handler with a projection reader.

        Args:
            projection_reader: Reader for querying registration projection state.
        """
        self._projection_reader = projection_reader

    @property
    def handler_id(self) -> str:
        """Return unique identifier for this handler."""
        return "handler-runtime-tick"

    @property
    def category(self) -> EnumMessageCategory:
        """Return the message category this handler processes."""
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Return the set of message types this handler processes."""
        return {"ModelRuntimeTick"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Return the node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> ModelHandlerOutput[object]:
        """Process runtime tick and emit timeout events.

        Scans the projection for overdue deadlines and emits appropriate
        timeout events. Uses projection emission markers to prevent
        duplicate timeout events.

        Args:
            envelope: The event envelope containing the runtime tick event.

        Returns:
            ModelHandlerOutput containing timeout events (ModelNodeRegistrationAckTimedOut,
            ModelNodeLivenessExpired). Events tuple may be empty if no timeouts detected.

        Raises:
            RuntimeHostError: If projection queries fail (propagated from reader).
            ProtocolConfigurationError: If envelope_timestamp is naive (no timezone info).
        """
        start_time = time.perf_counter()

        # Extract from envelope
        tick = envelope.payload
        now = envelope.envelope_timestamp
        correlation_id = envelope.correlation_id or uuid4()

        # Validate timezone-awareness for time injection pattern
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="handle_runtime_tick",
            target_name="handler.runtime_tick",
            correlation_id=correlation_id,
        )
        validate_timezone_aware_with_context(now, ctx)

        events: list[BaseModel] = []

        # 1. Check for overdue ack deadlines
        ack_timeout_events = await self._check_ack_timeouts(
            tick=tick,
            now=now,
            correlation_id=correlation_id,
        )
        events.extend(ack_timeout_events)

        # 2. Check for overdue liveness deadlines
        liveness_expired_events = await self._check_liveness_expiry(
            tick=tick,
            now=now,
            correlation_id=correlation_id,
        )
        events.extend(liveness_expired_events)

        if events:
            logger.info(
                "RuntimeTick processed, emitting timeout events",
                extra={
                    "tick_id": str(tick.tick_id),
                    "ack_timeout_count": len(ack_timeout_events),
                    "liveness_expired_count": len(liveness_expired_events),
                    "correlation_id": str(correlation_id),
                },
            )

        # Wrap result in ModelHandlerOutput
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=tuple(events),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )

    async def _check_ack_timeouts(
        self,
        tick: ModelRuntimeTick,
        now: datetime,
        correlation_id: UUID,
    ) -> list[ModelNodeRegistrationAckTimedOut]:
        """Check for entities with overdue ack deadlines.

        Queries the projection for entities in ack-waiting states
        (ACCEPTED, AWAITING_ACK) that have passed their ack_deadline
        and haven't had a timeout event emitted yet.

        Args:
            tick: The runtime tick event (used for causation_id).
            now: Current time for deadline comparison.
            correlation_id: Correlation ID for tracing.

        Returns:
            List of ModelNodeRegistrationAckTimedOut events to emit.
        """
        # Query projection for overdue ack registrations
        overdue_projections = (
            await self._projection_reader.get_overdue_ack_registrations(
                now=now,
                domain="registration",
                correlation_id=correlation_id,
            )
        )

        events: list[ModelNodeRegistrationAckTimedOut] = []

        for projection in overdue_projections:
            # Double-check with projection helper (defensive)
            if not projection.needs_ack_timeout_event(now):
                continue

            # Type narrowing: needs_ack_timeout_event() guarantees ack_deadline is not None
            ack_deadline = projection.ack_deadline
            if ack_deadline is None:
                # This should never happen - needs_ack_timeout_event() ensures ack_deadline
                # is not None. Log and skip this projection as a defensive measure.
                logger.warning(
                    "Skipping projection with None ack_deadline despite passing "
                    "needs_ack_timeout_event() check - this indicates a bug",
                    extra={
                        "entity_id": str(projection.entity_id),
                        "correlation_id": str(correlation_id),
                    },
                )
                continue

            event = ModelNodeRegistrationAckTimedOut(
                entity_id=projection.entity_id,
                node_id=projection.entity_id,
                correlation_id=correlation_id,
                causation_id=tick.tick_id,  # Link to triggering tick
                emitted_at=now,
                deadline_at=ack_deadline,
            )
            events.append(event)

            logger.info(
                "Detected ack timeout",
                extra={
                    "node_id": str(projection.entity_id),
                    "ack_deadline": (
                        projection.ack_deadline.isoformat()
                        if projection.ack_deadline
                        else None
                    ),
                    "correlation_id": str(correlation_id),
                },
            )

        return events

    async def _check_liveness_expiry(
        self,
        tick: ModelRuntimeTick,
        now: datetime,
        correlation_id: UUID,
    ) -> list[ModelNodeLivenessExpired]:
        """Check for active entities with overdue liveness deadlines.

        Queries the projection for ACTIVE entities that have passed
        their liveness_deadline and haven't had a liveness expired
        event emitted yet.

        Args:
            tick: The runtime tick event (used for causation_id).
            now: Current time for deadline comparison.
            correlation_id: Correlation ID for tracing.

        Returns:
            List of ModelNodeLivenessExpired events to emit.
        """
        # Query projection for overdue liveness registrations
        overdue_projections = (
            await self._projection_reader.get_overdue_liveness_registrations(
                now=now,
                domain="registration",
                correlation_id=correlation_id,
            )
        )

        events: list[ModelNodeLivenessExpired] = []

        for projection in overdue_projections:
            # Double-check with projection helper (defensive)
            if not projection.needs_liveness_timeout_event(now):
                continue

            # last_heartbeat_at semantic: None if no heartbeats were ever received.
            # This is intentionally different from registered_at - registration is
            # not a heartbeat. The projection tracks this field explicitly.
            event = ModelNodeLivenessExpired(
                entity_id=projection.entity_id,
                node_id=projection.entity_id,
                correlation_id=correlation_id,
                causation_id=tick.tick_id,  # Link to triggering tick
                emitted_at=now,
                last_heartbeat_at=projection.last_heartbeat_at,
            )
            events.append(event)

            logger.info(
                "Detected liveness expiry",
                extra={
                    "node_id": str(projection.entity_id),
                    "liveness_deadline": (
                        projection.liveness_deadline.isoformat()
                        if projection.liveness_deadline
                        else None
                    ),
                    "correlation_id": str(correlation_id),
                },
            )

        return events


__all__: list[str] = ["HandlerRuntimeTick"]

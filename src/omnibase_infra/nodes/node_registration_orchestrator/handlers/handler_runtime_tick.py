# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for RuntimeTick - timeout detection.

This handler processes RuntimeTick events from the runtime scheduler
and detects overdue ack and liveness deadlines. It queries the projection
for entities that need timeout events emitted, then delegates the decision
logic to RegistrationReducerService.decide_timeout().

Detection Logic:
    1. Query projection for overdue ack and liveness deadlines (I/O)
    2. Pass overdue projections to RegistrationReducerService.decide_timeout()
    3. The reducer decides which timeout events to emit (pure logic)
    4. Publish tombstones for liveness-expired nodes (best-effort I/O)

Deduplication (per C2 Durable Timeout Handling):
    The projection stores emission markers (ack_timeout_emitted_at,
    liveness_timeout_emitted_at) to prevent duplicate timeout events.
    The projection reader filters out already-emitted timeouts.
    The reducer performs a secondary deduplication check via
    projection.needs_*_timeout_event() helpers.

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
from omnibase_infra.models.registration.events.model_node_liveness_expired import (
    ModelNodeLivenessExpired,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_context import (
    ModelReducerContext,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.utils import (
    sanitize_error_message,
    validate_timezone_aware_with_context,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_snapshot_publisher import (
        ProtocolSnapshotPublisher,
    )

logger = logging.getLogger(__name__)


class HandlerRuntimeTick:
    """Handler for RuntimeTick - timeout detection.

    This handler processes runtime tick events by querying projections for
    overdue deadlines and delegating timeout decision logic to the
    RegistrationReducerService. The handler owns I/O (projection reads,
    snapshot tombstones); the reducer owns pure decision logic.

    Timeout Detection:
        On each tick the handler:
        1. Queries overdue ack and liveness projections (I/O)
        2. Calls reducer.decide_timeout() for event/intent decisions (pure)
        3. Publishes tombstones for liveness-expired nodes (best-effort I/O)

    Projection Queries:
        Uses dedicated projection reader methods that filter by:
        - Deadline < now (deadline has passed)
        - Emission marker IS NULL (not yet emitted)
        - Appropriate state (AWAITING_ACK for ack, ACTIVE for liveness)

    Attributes:
        _projection_reader: Reader for registration projection state.
        _reducer: Pure-function service for timeout decisions.
        _snapshot_publisher: Optional publisher for tombstone snapshots.

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
        >>> handler = HandlerRuntimeTick(projection_reader, reducer)
        >>> output = await handler.handle(envelope)
        >>> # Output events use injected `now` for emitted_at:
        >>> # ModelNodeRegistrationAckTimedOut(emitted_at=tick_time, ...)
        >>> # ModelNodeLivenessExpired(emitted_at=tick_time, last_heartbeat_at=<datetime|None>, ...)
        >>> # Note: last_heartbeat_at is None if no heartbeats were ever received
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        reducer: RegistrationReducerService,
        snapshot_publisher: ProtocolSnapshotPublisher | None = None,
    ) -> None:
        """Initialize the handler with a projection reader and reducer service.

        Args:
            projection_reader: Reader for querying registration projection state.
            reducer: Pure-function service for timeout decision logic.
            snapshot_publisher: Optional ProtocolSnapshotPublisher for publishing
                tombstones when nodes expire. If None, tombstone publishing is skipped.
                Tombstone publishing is always best-effort and non-blocking.
        """
        self._projection_reader = projection_reader
        self._reducer = reducer
        self._snapshot_publisher = snapshot_publisher

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

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler.

        Returns NODE_HANDLER because this handler processes node-level
        timeout detection events (not infrastructure plumbing).
        """
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns EFFECT because this handler performs side-effecting I/O:
        queries the registration projection in PostgreSQL to detect
        overdue ack and liveness deadlines.
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelRuntimeTick],
    ) -> ModelHandlerOutput[object]:
        """Process runtime tick and emit timeout events.

        Queries projections for overdue deadlines, delegates decision logic
        to the reducer, and publishes tombstones for expired nodes.

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

        # 1. Query overdue projections (I/O stays in handler)
        overdue_ack = await self._projection_reader.get_overdue_ack_registrations(
            now=now,
            domain="registration",
            correlation_id=correlation_id,
        )
        overdue_liveness = (
            await self._projection_reader.get_overdue_liveness_registrations(
                now=now,
                domain="registration",
                correlation_id=correlation_id,
            )
        )

        # 2. Reducer decides what events to emit
        reducer_ctx = ModelReducerContext(
            correlation_id=correlation_id,
            now=now,
            tick_id=tick.tick_id,
        )
        decision = self._reducer.decide_timeout(
            overdue_ack_projections=overdue_ack,
            overdue_liveness_projections=overdue_liveness,
            ctx=reducer_ctx,
        )

        if decision.action == "emit" and decision.events:
            logger.info(
                "RuntimeTick processed, emitting timeout events",
                extra={
                    "tick_id": str(tick.tick_id),
                    "event_count": len(decision.events),
                    "correlation_id": str(correlation_id),
                },
            )

        # 3. Publish tombstones for liveness-expired nodes (best-effort)
        if self._snapshot_publisher is not None:
            for event in decision.events:
                if isinstance(event, ModelNodeLivenessExpired):
                    try:
                        await self._snapshot_publisher.delete_snapshot(
                            str(event.entity_id), "registration"
                        )
                    except Exception as snap_err:
                        logger.warning(
                            "Snapshot tombstone publish failed (non-blocking): %s",
                            sanitize_error_message(snap_err),
                            extra={
                                "node_id": str(event.entity_id),
                                "correlation_id": str(correlation_id),
                                "error_type": type(snap_err).__name__,
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


__all__: list[str] = ["HandlerRuntimeTick"]

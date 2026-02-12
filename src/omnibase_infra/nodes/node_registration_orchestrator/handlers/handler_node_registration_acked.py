# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for NodeRegistrationAcked command - ack processing.

This handler processes NodeRegistrationAcked commands from nodes that
are acknowledging their registration. It queries the projection for
current state, delegates the decision to RegistrationReducerService.decide_ack(),
and applies the returned events, intents, and snapshot publishing.

The handler owns I/O (projection read, snapshot publish); the reducer
service owns pure decision logic (state checks, event/intent construction).

Processing Logic (delegated to RegistrationReducerService.decide_ack):
    If state is AWAITING_ACK or ACCEPTED:
        - Emit NodeRegistrationAckReceived + NodeBecameActive
        - Emit PostgreSQL UPDATE intent (state -> ACTIVE)
        - Publish ACTIVE snapshot (best-effort)

    If state is ACTIVE or ACK_RECEIVED:
        - Duplicate ack, no-op (idempotent)

    If state is terminal (REJECTED, LIVENESS_EXPIRED, ACK_TIMED_OUT):
        - Ack is too late, no-op

    If no projection exists:
        - Ack for unknown node, no-op

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different command instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumRegistrationState,
)
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.utils import (
    sanitize_error_message,
    validate_timezone_aware_with_context,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_snapshot_publisher import (
        ProtocolSnapshotPublisher,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OutputContext:
    """Context for creating handler output, bundling common parameters.

    This dataclass groups the parameters needed for creating ModelHandlerOutput,
    reducing the parameter count of _create_output from 6 to 3.
    """

    envelope: ModelEventEnvelope[ModelNodeRegistrationAcked]
    correlation_id: UUID
    now: datetime
    start_time: float


# Environment variable name for liveness interval configuration
ENV_LIVENESS_INTERVAL_SECONDS: Final[str] = "ONEX_LIVENESS_INTERVAL_SECONDS"

# Default liveness interval (60 seconds). This value is used when:
# 1. No explicit value is passed to the handler constructor
# 2. No environment variable ONEX_LIVENESS_INTERVAL_SECONDS is set
# 3. Container config does not specify liveness_interval_seconds
DEFAULT_LIVENESS_INTERVAL_SECONDS: Final[int] = 60


def get_liveness_interval_seconds(explicit_value: int | None = None) -> int:
    """Get liveness interval from explicit value, environment, or default.

    Resolution order (first non-None wins):
        1. Explicit value passed as parameter
        2. Environment variable ONEX_LIVENESS_INTERVAL_SECONDS
        3. Default constant (60 seconds)

    Args:
        explicit_value: Explicitly provided value (highest priority).
            Pass None to use environment or default.

    Returns:
        Liveness interval in seconds.

    Raises:
        ProtocolConfigurationError: If environment variable is set but not a valid integer.

    Example:
        >>> # Use default or env var
        >>> interval = get_liveness_interval_seconds()
        >>> # Force explicit value
        >>> interval = get_liveness_interval_seconds(120)
    """
    # 1. Explicit value takes priority
    if explicit_value is not None:
        return explicit_value

    # 2. Try environment variable
    env_value = os.getenv(ENV_LIVENESS_INTERVAL_SECONDS)
    if env_value is not None:
        try:
            return int(env_value)
        except ValueError as e:
            # Use ProtocolConfigurationError for invalid environment variable values.
            # No correlation_id available at module-level configuration, so context
            # is created without one (will auto-generate).
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="get_liveness_interval_seconds",
                target_name="env.ONEX_LIVENESS_INTERVAL_SECONDS",
            )
            raise ProtocolConfigurationError(
                f"Invalid value for {ENV_LIVENESS_INTERVAL_SECONDS}: "
                f"'{env_value}' is not a valid integer",
                context=ctx,
            ) from e

    # 3. Fall back to default
    return DEFAULT_LIVENESS_INTERVAL_SECONDS


class HandlerNodeRegistrationAcked:
    """Handler for NodeRegistrationAcked command - ack processing.

    This handler processes acknowledgment commands from nodes and
    decides whether to emit events that complete the registration
    workflow and activate the node.

    State Decision Matrix:
        | Current State       | Action                              |
        |---------------------|-------------------------------------|
        | None (unknown)      | No-op (warn: unknown node)          |
        | PENDING_REGISTRATION| No-op (ack too early, not accepted) |
        | ACCEPTED            | Emit AckReceived + BecameActive     |
        | AWAITING_ACK        | Emit AckReceived + BecameActive     |
        | ACK_RECEIVED        | No-op (duplicate, already received) |
        | ACTIVE              | No-op (duplicate, already active)   |
        | ACK_TIMED_OUT       | No-op (too late, timed out)         |
        | REJECTED            | No-op (terminal state)              |
        | LIVENESS_EXPIRED    | No-op (terminal state)              |

    Attributes:
        _projection_reader: Reader for registration projection state.
        _reducer: Pure-function reducer service for ack decision logic.
        _snapshot_publisher: Optional snapshot publisher for ACTIVE transitions.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
        >>> # Use explicit timestamps (time injection pattern) - not datetime.now()
        >>> now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        >>> envelope = ModelEventEnvelope(
        ...     payload=ack_command,
        ...     envelope_timestamp=now,
        ...     correlation_id=uuid4(),
        ... )
        >>> handler = HandlerNodeRegistrationAcked(projection_reader, reducer)
        >>> output = await handler.handle(envelope)
        >>> # output.events may contain (AckReceived, BecameActive)
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
            reducer: Pure-function reducer service that encapsulates the ack
                decision logic (state checks, event/intent creation).
            snapshot_publisher: Optional ProtocolSnapshotPublisher for publishing
                compacted snapshots to Kafka after ACTIVE transition. If None,
                snapshot publishing is skipped. Snapshot publishing is always
                best-effort and non-blocking.
        """
        self._projection_reader = projection_reader
        self._reducer = reducer
        self._snapshot_publisher = snapshot_publisher

    @property
    def handler_id(self) -> str:
        """Return the unique identifier for this handler."""
        return "handler-node-registration-acked"

    @property
    def category(self) -> EnumMessageCategory:
        """Return the message category this handler processes."""
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        """Return the set of message type names this handler processes."""
        return {"ModelNodeRegistrationAcked"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Return the node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler.

        Returns NODE_HANDLER because this handler processes node-level
        registration acknowledgment commands (not infrastructure plumbing).
        """
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns EFFECT because this handler performs side-effecting I/O:
        reads projection state from PostgreSQL and publishes snapshots.
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelNodeRegistrationAcked],
    ) -> ModelHandlerOutput[object]:
        """Process registration ack command and emit events.

        Queries the current projection state and decides whether to
        emit events that complete registration and activate the node.

        Args:
            envelope: The event envelope containing the registration ack command.

        Returns:
            ModelHandlerOutput containing [NodeRegistrationAckReceived, NodeBecameActive]
            if ack is valid, empty events tuple otherwise.

        Raises:
            RuntimeHostError: If projection query fails (propagated from reader).
            ProtocolConfigurationError: If timestamp is naive (no timezone info).
        """
        start_time = time.perf_counter()

        # Extract from envelope
        command = envelope.payload
        now = envelope.envelope_timestamp
        correlation_id = envelope.correlation_id or uuid4()

        # Validate timezone-awareness for time injection pattern
        error_ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="handle_registration_acked",
            target_name="handler.node_registration_acked",
            correlation_id=correlation_id,
        )
        validate_timezone_aware_with_context(now, error_ctx)

        # Create output context for _create_output calls
        ctx = OutputContext(
            envelope=envelope,
            correlation_id=correlation_id,
            now=now,
            start_time=start_time,
        )

        node_id = command.node_id

        # Query current projection state
        projection = await self._projection_reader.get_entity_state(
            entity_id=node_id,
            domain="registration",
            correlation_id=correlation_id,
        )

        # Delegate state decision to the pure-function reducer service
        decision = self._reducer.decide_ack(
            projection=projection,
            command=command,
            correlation_id=correlation_id,
            now=now,
        )

        if decision.action == "no_op":
            logger.debug(
                "ACK decision: no_op",
                extra={
                    "node_id": str(node_id),
                    "reason": decision.reason,
                    "correlation_id": str(correlation_id),
                },
            )
            return self._create_output(ctx=ctx, events=())

        # Publish snapshot after ACTIVE transition (best-effort, non-blocking).
        # We construct a synthetic projection with the post-transition state
        # (ACTIVE) because `projection` still holds the pre-transition state
        # (AWAITING_ACK/ACCEPTED). The emitted events will transition the
        # state in the reducer, but the snapshot must reflect the target state.
        if (
            self._snapshot_publisher is not None
            and projection is not None
            and decision.new_state == EnumRegistrationState.ACTIVE
        ):
            try:
                active_projection = projection.model_copy(
                    update={
                        "current_state": EnumRegistrationState.ACTIVE,
                        "updated_at": now,
                    }
                )
                # node_name is None because neither the ack command nor the
                # registration projection carries a human-readable node name.
                # The introspection event that originally provided the name is
                # not available in this handler's scope.  The snapshot consumer
                # can resolve it via entity_id if needed.
                await self._snapshot_publisher.publish_from_projection(
                    active_projection,
                    node_name=None,
                    correlation_id=correlation_id,
                )
            except Exception as snap_err:
                logger.warning(
                    "Snapshot publish failed (non-blocking): %s",
                    sanitize_error_message(snap_err),
                    extra={
                        "node_id": str(node_id),
                        "correlation_id": str(correlation_id),
                        "error_type": type(snap_err).__name__,
                    },
                )

        return self._create_output(
            ctx=ctx, events=decision.events, intents=decision.intents
        )

    def _create_output(
        self,
        ctx: OutputContext,
        events: tuple[BaseModel, ...],
        intents: tuple[ModelIntent, ...] = (),
    ) -> ModelHandlerOutput[object]:
        """Create a ModelHandlerOutput with the given events and intents.

        Args:
            ctx: Output context containing envelope, correlation_id, now, start_time.
            events: Tuple of events to include in the output.
            intents: Tuple of intents for the effect layer (default empty).

        Returns:
            ModelHandlerOutput with the provided events, intents, and metadata.
        """
        processing_time_ms = (time.perf_counter() - ctx.start_time) * 1000
        return ModelHandlerOutput(
            input_envelope_id=ctx.envelope.envelope_id,
            correlation_id=ctx.correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=events,
            intents=intents,
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=ctx.now,
        )


__all__: list[str] = [
    "DEFAULT_LIVENESS_INTERVAL_SECONDS",
    "ENV_LIVENESS_INTERVAL_SECONDS",
    "HandlerNodeRegistrationAcked",
    "get_liveness_interval_seconds",
]

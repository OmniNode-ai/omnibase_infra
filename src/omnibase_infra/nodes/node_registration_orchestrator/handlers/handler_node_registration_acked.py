# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for NodeRegistrationAcked command - ack processing.

This handler processes NodeRegistrationAcked commands from nodes that
are acknowledging their registration. It queries the projection for
current state and emits appropriate events.

Processing Logic:
    If state is AWAITING_ACK:
        - Emit NodeRegistrationAckReceived
        - Emit NodeBecameActive (with capabilities snapshot)
        - Set liveness_deadline for heartbeat monitoring

    If state is ACTIVE:
        - Duplicate ack, no-op (idempotent)

    If state is terminal (REJECTED, LIVENESS_EXPIRED):
        - Ack is too late, no-op (log warning)

    If no projection exists:
        - Ack for unknown node, no-op (log warning)

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
from datetime import datetime, timedelta
from typing import Final
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.events.model_node_became_active import (
    ModelNodeBecameActive,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_received import (
    ModelNodeRegistrationAckReceived,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.utils import validate_timezone_aware_with_context

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
        _liveness_interval_seconds: Interval for liveness deadline.

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
        >>> handler = HandlerNodeRegistrationAcked(projection_reader)
        >>> output = await handler.handle(envelope)
        >>> # output.events may contain (AckReceived, BecameActive)
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        liveness_interval_seconds: int | None = None,
    ) -> None:
        """Initialize the handler with a projection reader.

        Args:
            projection_reader: Reader for querying registration projection state.
            liveness_interval_seconds: Interval for liveness deadline calculation.
                Pass None to use environment variable ONEX_LIVENESS_INTERVAL_SECONDS
                or default (60 seconds).
        """
        self._projection_reader = projection_reader
        self._liveness_interval_seconds = get_liveness_interval_seconds(
            liveness_interval_seconds
        )

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

        # Decision: Is this a valid ack?
        if projection is None:
            # Unknown node - ack for non-existent registration
            logger.warning(
                "Received ack for unknown node",
                extra={
                    "node_id": str(node_id),
                    "correlation_id": str(correlation_id),
                },
            )
            return self._create_output(ctx=ctx, events=())

        current_state = projection.current_state

        # Check if ack is valid for current state
        if current_state in {
            EnumRegistrationState.ACCEPTED,
            EnumRegistrationState.AWAITING_ACK,
        }:
            # Valid ack - emit events
            events = self._emit_activation_events(
                command=command,
                now=now,
                correlation_id=correlation_id,
                projection=projection,
            )
            return self._create_output(ctx=ctx, events=tuple(events))

        # Handle other states
        if current_state in {
            EnumRegistrationState.ACK_RECEIVED,
            EnumRegistrationState.ACTIVE,
        }:
            # Duplicate ack - idempotent no-op
            logger.debug(
                "Duplicate ack received, ignoring",
                extra={
                    "node_id": str(node_id),
                    "current_state": str(current_state),
                    "correlation_id": str(correlation_id),
                },
            )
            return self._create_output(ctx=ctx, events=())

        if current_state == EnumRegistrationState.PENDING_REGISTRATION:
            # Ack too early - not yet accepted
            logger.warning(
                "Ack received before registration accepted",
                extra={
                    "node_id": str(node_id),
                    "current_state": str(current_state),
                    "correlation_id": str(correlation_id),
                },
            )
            return self._create_output(ctx=ctx, events=())

        if current_state == EnumRegistrationState.ACK_TIMED_OUT:
            # Ack too late - already timed out
            logger.warning(
                "Ack received after timeout",
                extra={
                    "node_id": str(node_id),
                    "current_state": str(current_state),
                    "correlation_id": str(correlation_id),
                },
            )
            return self._create_output(ctx=ctx, events=())

        if current_state.is_terminal():
            # Terminal state - ack is meaningless
            logger.warning(
                "Ack received for node in terminal state",
                extra={
                    "node_id": str(node_id),
                    "current_state": str(current_state),
                    "correlation_id": str(correlation_id),
                },
            )
            return self._create_output(ctx=ctx, events=())

        # Unexpected state - log and return empty
        logger.warning(
            "Ack received for node in unexpected state",
            extra={
                "node_id": str(node_id),
                "current_state": str(current_state),
                "correlation_id": str(correlation_id),
            },
        )
        return self._create_output(ctx=ctx, events=())

    def _emit_activation_events(
        self,
        command: ModelNodeRegistrationAcked,
        now: datetime,
        correlation_id: UUID,
        projection: ModelRegistrationProjection,
    ) -> list[BaseModel]:
        """Emit events for successful registration acknowledgment.

        Creates and returns the events that represent the node becoming
        active after successful ack.

        Args:
            command: The registration ack command.
            now: Current time for liveness deadline calculation.
            correlation_id: Correlation ID for tracing.
            projection: Current projection state (for capabilities).

        Returns:
            List containing [NodeRegistrationAckReceived, NodeBecameActive].
        """

        node_id = command.node_id
        liveness_deadline = now + timedelta(seconds=self._liveness_interval_seconds)

        # Event 1: Ack received
        ack_received = ModelNodeRegistrationAckReceived(
            entity_id=node_id,
            node_id=node_id,
            correlation_id=correlation_id,
            causation_id=command.command_id,
            emitted_at=now,  # Use injected time for consistency
            liveness_deadline=liveness_deadline,
        )

        # Event 2: Node became active
        became_active = ModelNodeBecameActive(
            entity_id=node_id,
            node_id=node_id,
            correlation_id=correlation_id,
            causation_id=command.command_id,
            emitted_at=now,  # Use injected time for consistency
            capabilities=projection.capabilities,
        )

        logger.info(
            "Emitting activation events",
            extra={
                "node_id": str(node_id),
                "liveness_deadline": liveness_deadline.isoformat(),
                "correlation_id": str(correlation_id),
            },
        )

        return [ack_received, became_active]

    def _create_output(
        self,
        ctx: OutputContext,
        events: tuple[BaseModel, ...],
    ) -> ModelHandlerOutput[object]:
        """Create a ModelHandlerOutput with the given events.

        Args:
            ctx: Output context containing envelope, correlation_id, now, start_time.
            events: Tuple of events to include in the output.

        Returns:
            ModelHandlerOutput with the provided events and metadata.
        """
        processing_time_ms = (time.perf_counter() - ctx.start_time) * 1000
        return ModelHandlerOutput(
            input_envelope_id=ctx.envelope.envelope_id,
            correlation_id=ctx.correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=events,
            intents=(),
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

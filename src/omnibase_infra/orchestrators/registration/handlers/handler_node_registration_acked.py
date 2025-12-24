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

Thread Safety:
    This handler is stateless and thread-safe for concurrent calls
    with different command instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
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

logger = logging.getLogger(__name__)


# TODO(OMN-XXX): Make liveness interval configurable via container
# For MVP, using reasonable default of 60 seconds
_DEFAULT_LIVENESS_INTERVAL_SECONDS: int = 60


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
        >>> handler = HandlerNodeRegistrationAcked(projection_reader)
        >>> events = await handler.handle(
        ...     command=ack_command,
        ...     now=datetime.now(UTC),
        ...     correlation_id=uuid4(),
        ... )
        >>> # events may contain [AckReceived, BecameActive]
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        liveness_interval_seconds: int = _DEFAULT_LIVENESS_INTERVAL_SECONDS,
    ) -> None:
        """Initialize the handler with a projection reader.

        Args:
            projection_reader: Reader for querying registration projection state.
            liveness_interval_seconds: Interval for liveness deadline calculation.
                                       Defaults to 60 seconds.
        """
        self._projection_reader = projection_reader
        self._liveness_interval_seconds = liveness_interval_seconds

    async def handle(
        self,
        command: ModelNodeRegistrationAcked,
        now: datetime,
        correlation_id: UUID,
    ) -> list[BaseModel]:
        """Process registration ack command and emit events.

        Queries the current projection state and decides whether to
        emit events that complete registration and activate the node.

        Args:
            command: The registration ack command from the node.
            now: Injected current time for liveness deadline calculation.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            List containing [NodeRegistrationAckReceived, NodeBecameActive]
            if ack is valid, empty list otherwise.

        Raises:
            RuntimeHostError: If projection query fails (propagated from reader).
        """
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
            return []

        current_state = projection.current_state

        # Check if ack is valid for current state
        if current_state in {
            EnumRegistrationState.ACCEPTED,
            EnumRegistrationState.AWAITING_ACK,
        }:
            # Valid ack - emit events
            return self._emit_activation_events(
                command=command,
                now=now,
                correlation_id=correlation_id,
                projection=projection,
            )

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
            return []

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
            return []

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
            return []

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
            return []

        # Unexpected state - log and return empty
        logger.warning(
            "Ack received for node in unexpected state",
            extra={
                "node_id": str(node_id),
                "current_state": str(current_state),
                "correlation_id": str(correlation_id),
            },
        )
        return []

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


__all__: list[str] = ["HandlerNodeRegistrationAcked"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for NodeIntrospectionEvent - canonical registration trigger.

This handler processes NodeIntrospectionEvent payloads from nodes announcing
their presence in the cluster. It queries the projection for current state
and emits NodeRegistrationInitiated if the node is new or needs to retry.

Decision Logic:
    The handler emits NodeRegistrationInitiated when:
    - No projection exists (new node)
    - State is LIVENESS_EXPIRED (re-registration after death)
    - State is REJECTED (retry after rejection)
    - State is ACK_TIMED_OUT (retry after timeout)

    The handler does NOT emit when:
    - State is PENDING_REGISTRATION (already processing)
    - State is ACCEPTED (already accepted, waiting for ack)
    - State is AWAITING_ACK (already waiting for ack)
    - State is ACK_RECEIVED (already acknowledged)
    - State is ACTIVE (already active - heartbeat should be used)

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different event instances.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-944 (F1): Registration Projection Schema
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumRegistrationState

if TYPE_CHECKING:
    from pydantic import BaseModel
from omnibase_infra.models.registration.events.model_node_registration_initiated import (
    ModelNodeRegistrationInitiated,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)

logger = logging.getLogger(__name__)


# States that allow re-registration (node can try again)
_RETRIABLE_STATES: frozenset[EnumRegistrationState] = frozenset(
    {
        EnumRegistrationState.LIVENESS_EXPIRED,
        EnumRegistrationState.REJECTED,
        EnumRegistrationState.ACK_TIMED_OUT,
    }
)

# States that block new registration (already in progress or active)
_BLOCKING_STATES: frozenset[EnumRegistrationState] = frozenset(
    {
        EnumRegistrationState.PENDING_REGISTRATION,
        EnumRegistrationState.ACCEPTED,
        EnumRegistrationState.AWAITING_ACK,
        EnumRegistrationState.ACK_RECEIVED,
        EnumRegistrationState.ACTIVE,
    }
)


class HandlerNodeIntrospected:
    """Handler for NodeIntrospectionEvent - canonical registration trigger.

    This handler processes introspection events from nodes announcing
    themselves to the cluster. It queries the current projection state
    and decides whether to initiate a new registration workflow.

    State Decision Matrix:
        | Current State       | Action                          |
        |---------------------|----------------------------------|
        | None (new node)     | Emit NodeRegistrationInitiated   |
        | LIVENESS_EXPIRED    | Emit NodeRegistrationInitiated   |
        | REJECTED            | Emit NodeRegistrationInitiated   |
        | ACK_TIMED_OUT       | Emit NodeRegistrationInitiated   |
        | PENDING_REGISTRATION| No-op (already processing)       |
        | ACCEPTED            | No-op (waiting for ack)          |
        | AWAITING_ACK        | No-op (waiting for ack)          |
        | ACK_RECEIVED        | No-op (transitioning to active)  |
        | ACTIVE              | No-op (use heartbeat instead)    |

    Attributes:
        _projection_reader: Reader for registration projection state.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> # Use explicit timestamps (time injection pattern) - not datetime.now()
        >>> now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        >>> handler = HandlerNodeIntrospected(projection_reader)
        >>> events = await handler.handle(
        ...     event=introspection_event,
        ...     now=now,
        ...     correlation_id=uuid4(),
        ... )
        >>> if events:
        ...     assert isinstance(events[0], ModelNodeRegistrationInitiated)
    """

    def __init__(self, projection_reader: ProjectionReaderRegistration) -> None:
        """Initialize the handler with a projection reader.

        Args:
            projection_reader: Reader for querying registration projection state.
        """
        self._projection_reader = projection_reader

    async def handle(
        self,
        event: ModelNodeIntrospectionEvent,
        now: datetime,
        correlation_id: UUID,
    ) -> list[BaseModel]:
        """Process introspection event and decide on registration.

        Queries the current projection state for the node and decides
        whether to emit a NodeRegistrationInitiated event to start
        the registration workflow.

        Args:
            event: The introspection event from the node.
            now: Injected current time (for consistency, not used in decision).
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            List containing ModelNodeRegistrationInitiated if registration
            should be initiated, empty list otherwise.

        Raises:
            RuntimeHostError: If projection query fails (propagated from reader).
        """
        node_id = event.node_id

        # Query current projection state
        projection = await self._projection_reader.get_entity_state(
            entity_id=node_id,
            domain="registration",
            correlation_id=correlation_id,
        )

        # Decision: Should we initiate registration?
        should_initiate = False
        current_state: EnumRegistrationState | None = None

        if projection is None:
            # New node - initiate registration
            should_initiate = True
            logger.info(
                "New node detected, initiating registration",
                extra={
                    "node_id": str(node_id),
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            current_state = projection.current_state

            if current_state in _RETRIABLE_STATES:
                # Retriable state - allow re-registration
                should_initiate = True
                logger.info(
                    "Node in retriable state, initiating re-registration",
                    extra={
                        "node_id": str(node_id),
                        "current_state": str(current_state),
                        "correlation_id": str(correlation_id),
                    },
                )
            elif current_state in _BLOCKING_STATES:
                # Blocking state - no-op
                should_initiate = False
                logger.debug(
                    "Node in blocking state, skipping registration",
                    extra={
                        "node_id": str(node_id),
                        "current_state": str(current_state),
                        "correlation_id": str(correlation_id),
                    },
                )

        if not should_initiate:
            return []

        # Emit NodeRegistrationInitiated
        initiated_event = ModelNodeRegistrationInitiated(
            entity_id=node_id,
            node_id=node_id,
            correlation_id=correlation_id,
            causation_id=event.correlation_id,  # Link to triggering event
            emitted_at=now,  # Use injected time for consistency
            registration_attempt_id=uuid4(),
        )

        logger.info(
            "Emitting NodeRegistrationInitiated",
            extra={
                "node_id": str(node_id),
                "registration_attempt_id": str(initiated_event.registration_attempt_id),
                "correlation_id": str(correlation_id),
            },
        )

        return [initiated_event]


__all__: list[str] = ["HandlerNodeIntrospected"]

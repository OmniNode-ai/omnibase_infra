# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Heartbeat Handler for Registration Orchestrator.

Processes NodeHeartbeatReceived events and updates the registration projection
with `last_heartbeat_at` and extended `liveness_deadline`.

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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent

if TYPE_CHECKING:
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime.projector_shell import ProjectorShell

logger = logging.getLogger(__name__)

# Default liveness window in seconds (matches mixin_node_introspection heartbeat interval)
DEFAULT_LIVENESS_WINDOW_SECONDS: float = 90.0


class ModelHeartbeatHandlerResult(BaseModel):
    """Result model for heartbeat handler processing.

    Attributes:
        success: Whether the heartbeat was processed successfully.
        node_id: UUID of the node that sent the heartbeat.
        previous_state: The node's state before processing (if found).
        last_heartbeat_at: Updated heartbeat timestamp.
        liveness_deadline: Extended liveness deadline.
        node_not_found: True if no projection exists for this node.
        correlation_id: Correlation ID for distributed tracing.
        error_message: Error message if processing failed (success=False).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    success: bool = Field(
        ...,
        description="Whether the heartbeat was processed successfully",
    )
    node_id: UUID = Field(
        ...,
        description="UUID of the node that sent the heartbeat",
    )
    previous_state: EnumRegistrationState | None = Field(
        default=None,
        description="The node's state before processing (if found)",
    )
    last_heartbeat_at: datetime | None = Field(
        default=None,
        description="Updated heartbeat timestamp",
    )
    liveness_deadline: datetime | None = Field(
        default=None,
        description="Extended liveness deadline",
    )
    node_not_found: bool = Field(
        default=False,
        description="True if no projection exists for this node",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if processing failed",
    )


class HandlerNodeHeartbeat:
    """Handler for processing node heartbeat events.

    Processes ModelNodeHeartbeatEvent events and updates the registration
    projection with:
    - `last_heartbeat_at`: Set to event timestamp (or current time)
    - `liveness_deadline`: Extended by liveness_window_seconds from now

    The handler requires both a projection reader (for lookups) and a projector
    (for updates). It is designed to be used by the registration orchestrator.

    ONEX Contract Compliance:
        This handler belongs to an ORCHESTRATOR node, so it returns result=None
        per ONEX contract rules (ORCHESTRATOR nodes use events[] and intents[]
        only, not result). Callers should verify success by querying the
        projection state after handle() returns.

    Error Handling:
        - If no projection exists, logs warning and returns empty output
        - Only ACTIVE nodes should receive heartbeats; other states log warnings
        - Database errors are re-raised as InfraConnectionError/InfraTimeoutError

    Coroutine Safety:
        This handler is stateless and coroutine-safe. The projection reader and
        projector are assumed to be coroutine-safe (they use connection pools).

    Example:
        >>> from omnibase_infra.projectors import ProjectionReaderRegistration
        >>> from omnibase_infra.runtime.projector_shell import ProjectorShell
        >>> handler = HandlerNodeHeartbeat(
        ...     projection_reader=reader,
        ...     projector=projector,
        ...     liveness_window_seconds=90.0,
        ... )
        >>> output = await handler.handle(envelope)
        >>> # Verify success by checking projection state
        >>> projection = await reader.get_entity_state(node_id)
        >>> if projection and projection.last_heartbeat_at == event.timestamp:
        ...     print(f"Heartbeat processed, deadline: {projection.liveness_deadline}")
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        projector: ProjectorShell,
        liveness_window_seconds: float = DEFAULT_LIVENESS_WINDOW_SECONDS,
    ) -> None:
        """Initialize the heartbeat handler.

        Args:
            projection_reader: Projection reader for looking up node state.
            projector: ProjectorShell for persisting heartbeat updates.
                Should be loaded from the registration projector contract.
            liveness_window_seconds: How long to extend liveness_deadline from
                the heartbeat timestamp. Default: 90 seconds (3x the default
                30-second heartbeat interval, allowing for 2 missed heartbeats).
        """
        self._projection_reader = projection_reader
        self._projector = projector
        self._liveness_window_seconds = liveness_window_seconds

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
    def liveness_window_seconds(self) -> float:
        """Return configured liveness window in seconds."""
        return self._liveness_window_seconds

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelNodeHeartbeatEvent],
    ) -> ModelHandlerOutput[object]:
        """Process a node heartbeat event.

        Looks up the registration projection by node_id and updates:
        - `last_heartbeat_at`: Set to event.timestamp
        - `liveness_deadline`: Extended to event.timestamp + liveness_window

        ONEX Contract Compliance:
            This handler belongs to an ORCHESTRATOR node, so it returns
            result=None per ONEX contract rules. Success/failure should be
            verified by querying the projection state after handle() returns.

        Args:
            envelope: Event envelope containing the heartbeat event payload.

        Returns:
            ModelHandlerOutput with result=None. Check projection state via
            ProjectionReaderRegistration.get_entity_state() to verify heartbeat
            was processed successfully.

        Raises:
            RuntimeHostError: Base class for all infrastructure errors. Specific
                subclasses are preserved and can be caught by callers:
                - InfraConnectionError: Database connection failures
                - InfraTimeoutError: Operation timeout exceeded
                - InfraAuthenticationError: Authentication/authorization failures
                - InfraUnavailableError: Resource temporarily unavailable

        Example:
            >>> output = await handler.handle(envelope)
            >>> # Verify success by checking projection state
            >>> projection = await reader.get_entity_state(node_id)
            >>> if projection and projection.last_heartbeat_at == event.timestamp:
            ...     print("Heartbeat processed successfully")
        """
        start_time = time.perf_counter()

        # Extract from envelope
        event = envelope.payload
        now = envelope.envelope_timestamp
        correlation_id = envelope.correlation_id or uuid4()
        domain = "registration"  # Was passed as parameter, now hardcoded

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="handle_heartbeat",
            target_name="handler.node_heartbeat",
            correlation_id=correlation_id,
        )

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

        # Calculate new liveness deadline
        heartbeat_timestamp = event.timestamp
        new_liveness_deadline = heartbeat_timestamp + timedelta(
            seconds=self._liveness_window_seconds
        )

        # Update projection via projector using partial_update
        try:
            updated = await self._projector.partial_update(
                aggregate_id=event.node_id,
                updates={
                    "last_heartbeat_at": heartbeat_timestamp,
                    "liveness_deadline": new_liveness_deadline,
                    "updated_at": now,
                },
                correlation_id=correlation_id,
            )

            if not updated:
                # Entity was not found (unlikely since we just read it, but handle it)
                logger.warning(
                    "Failed to update heartbeat - entity not found during update",
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

            logger.debug(
                "Heartbeat processed successfully",
                extra={
                    "node_id": str(event.node_id),
                    "last_heartbeat_at": heartbeat_timestamp.isoformat(),
                    "liveness_deadline": new_liveness_deadline.isoformat(),
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

        except ModelOnexError:
            # Re-raise all ONEX errors directly (preserves error type)
            # This includes:
            # - RuntimeHostError and all its subclasses (InfraConnectionError, etc.)
            # - ModelOnexError raised directly by other ONEX components
            # Callers can catch specific types for differentiated handling
            raise
        except Exception as e:
            # Wrap only non-infrastructure errors in RuntimeHostError
            # This should be rare - most errors from projector are infrastructure errors
            logger.exception(
                "Unexpected error updating heartbeat",
                extra={
                    "node_id": str(event.node_id),
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeHostError(
                f"Unexpected error updating heartbeat: {type(e).__name__}",
                context=ctx,
            ) from e


__all__ = [
    "DEFAULT_LIVENESS_WINDOW_SECONDS",
    "HandlerNodeHeartbeat",
    "ModelHeartbeatHandlerResult",
]

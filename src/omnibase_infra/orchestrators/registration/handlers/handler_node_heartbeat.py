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
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent

if TYPE_CHECKING:
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )

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

    Error Handling:
        - Returns node_not_found=True if no projection exists for the node
        - Only ACTIVE nodes should receive heartbeats; other states log warnings
        - Database errors are re-raised as InfraConnectionError/InfraTimeoutError

    Thread Safety:
        This handler is stateless and thread-safe. The projection reader and
        projector are assumed to be thread-safe (they use connection pools).

    Example:
        >>> from omnibase_infra.projectors import (
        ...     ProjectionReaderRegistration,
        ...     ProjectorRegistration,
        ... )
        >>> handler = HandlerNodeHeartbeat(
        ...     projection_reader=reader,
        ...     projector=projector,
        ...     liveness_window_seconds=90.0,
        ... )
        >>> result = await handler.handle(heartbeat_event)
        >>> if result.success:
        ...     print(f"Heartbeat processed, deadline extended to {result.liveness_deadline}")
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
        liveness_window_seconds: float = DEFAULT_LIVENESS_WINDOW_SECONDS,
    ) -> None:
        """Initialize the heartbeat handler.

        Args:
            projection_reader: Projection reader for looking up node state.
            projector: Projector for persisting heartbeat updates.
            liveness_window_seconds: How long to extend liveness_deadline from
                the heartbeat timestamp. Default: 90 seconds (3x the default
                30-second heartbeat interval, allowing for 2 missed heartbeats).
        """
        self._projection_reader = projection_reader
        self._projector = projector
        self._liveness_window_seconds = liveness_window_seconds

    @property
    def liveness_window_seconds(self) -> float:
        """Return configured liveness window in seconds."""
        return self._liveness_window_seconds

    async def handle(
        self,
        event: ModelNodeHeartbeatEvent,
        domain: str = "registration",
    ) -> ModelHeartbeatHandlerResult:
        """Process a node heartbeat event.

        Looks up the registration projection by node_id and updates:
        - `last_heartbeat_at`: Set to event.timestamp
        - `liveness_deadline`: Extended to event.timestamp + liveness_window

        Args:
            event: The heartbeat event to process.
            domain: Domain namespace for projection lookup (default: "registration").

        Returns:
            ModelHeartbeatHandlerResult with processing outcome.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If database operation times out.
            RuntimeHostError: For other infrastructure errors.

        Example:
            >>> result = await handler.handle(heartbeat_event)
            >>> if result.node_not_found:
            ...     logger.warning("Heartbeat from unregistered node")
        """
        correlation_id = event.correlation_id or uuid4()
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
            return ModelHeartbeatHandlerResult(
                success=False,
                node_id=event.node_id,
                node_not_found=True,
                correlation_id=correlation_id,
                error_message="No registration projection found for node",
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

        # Update projection via projector
        try:
            updated = await self._projector.update_heartbeat(
                entity_id=event.node_id,
                domain=domain,
                last_heartbeat_at=heartbeat_timestamp,
                liveness_deadline=new_liveness_deadline,
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
                return ModelHeartbeatHandlerResult(
                    success=False,
                    node_id=event.node_id,
                    previous_state=projection.current_state,
                    node_not_found=True,
                    correlation_id=correlation_id,
                    error_message="Entity not found during heartbeat update",
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

            return ModelHeartbeatHandlerResult(
                success=True,
                node_id=event.node_id,
                previous_state=projection.current_state,
                last_heartbeat_at=heartbeat_timestamp,
                liveness_deadline=new_liveness_deadline,
                correlation_id=correlation_id,
            )

        except Exception as e:
            logger.exception(
                "Failed to update heartbeat",
                extra={
                    "node_id": str(event.node_id),
                    "correlation_id": str(correlation_id),
                },
            )
            raise RuntimeHostError(
                f"Failed to update heartbeat: {type(e).__name__}",
                context=ctx,
            ) from e


__all__ = [
    "HandlerNodeHeartbeat",
    "ModelHeartbeatHandlerResult",
    "DEFAULT_LIVENESS_WINDOW_SECONDS",
]

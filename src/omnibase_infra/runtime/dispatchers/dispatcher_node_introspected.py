# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dispatcher adapter for HandlerNodeIntrospected.

This module provides a ProtocolMessageDispatcher adapter that wraps
HandlerNodeIntrospected for integration with MessageDispatchEngine.

The adapter:
- Deserializes ModelEventEnvelope payload to ModelNodeIntrospectionEvent
- Extracts correlation_id from envelope metadata
- Injects current time via ModelDispatchContext (for ORCHESTRATOR node kind)
- Calls the wrapped handler and emits output events

Design:
    The adapter follows ONEX dispatcher patterns:
    - Implements ProtocolMessageDispatcher protocol
    - Stateless operation (handler instance is injected)
    - Returns ModelDispatchResult with success/failure status
    - Uses EnumNodeKind.ORCHESTRATOR for time injection

Related:
    - OMN-888: Registration Orchestrator
    - OMN-892: 2-way Registration E2E Integration Test
"""

from __future__ import annotations

__all__ = ["DispatcherNodeIntrospected"]

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
    )

logger = logging.getLogger(__name__)


class DispatcherNodeIntrospected:
    """Dispatcher adapter for HandlerNodeIntrospected.

    This dispatcher wraps HandlerNodeIntrospected to integrate it with
    MessageDispatchEngine's category-based routing. It handles:

    - Deserialization: Validates and casts envelope payload to ModelNodeIntrospectionEvent
    - Time injection: Uses current time from dispatch context
    - Correlation tracking: Extracts or generates correlation_id
    - Error handling: Returns structured ModelDispatchResult on failure

    Thread Safety:
        This dispatcher is stateless and safe for concurrent invocation.
        The wrapped handler must also be coroutine-safe.

    Attributes:
        _handler: The wrapped HandlerNodeIntrospected instance.

    Example:
        >>> from omnibase_infra.runtime.dispatchers import DispatcherNodeIntrospected
        >>> dispatcher = DispatcherNodeIntrospected(handler_instance)
        >>> result = await dispatcher.handle(envelope)
    """

    def __init__(self, handler: HandlerNodeIntrospected) -> None:
        """Initialize dispatcher with wrapped handler.

        Args:
            handler: HandlerNodeIntrospected instance to delegate to.
        """
        self._handler = handler

    @property
    def dispatcher_id(self) -> str:
        """Unique identifier for this dispatcher.

        Returns:
            str: The dispatcher ID used for registration and tracing.
        """
        return "dispatcher.registration.node-introspected"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this dispatcher processes.

        Returns:
            EnumMessageCategory: EVENT category (introspection events).
        """
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Specific message types this dispatcher accepts.

        Returns:
            set[str]: Set containing ModelNodeIntrospectionEvent type name.
        """
        return {"ModelNodeIntrospectionEvent"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """ONEX node kind for time injection rules.

        Returns:
            EnumNodeKind: ORCHESTRATOR for workflow coordination with time.
        """
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self,
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult:
        """Handle introspection event and return dispatch result.

        Deserializes the envelope payload to ModelNodeIntrospectionEvent,
        delegates to the wrapped handler, and returns a structured result.

        Args:
            envelope: Event envelope containing introspection payload.

        Returns:
            ModelDispatchResult: Success with output events or error details.
        """
        started_at = datetime.now(UTC)
        correlation_id = envelope.correlation_id or uuid4()

        try:
            # Validate payload type
            payload = envelope.payload
            if not isinstance(payload, ModelNodeIntrospectionEvent):
                # Try to construct from dict if payload is dict-like
                if isinstance(payload, dict):
                    payload = ModelNodeIntrospectionEvent.model_validate(payload)
                else:
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic="node.introspection",
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                        duration_ms=(datetime.now(UTC) - started_at).total_seconds()
                        * 1000,
                        error_message=f"Expected ModelNodeIntrospectionEvent payload, "
                        f"got {type(payload).__name__}",
                        correlation_id=correlation_id,
                    )

            # Get current time for handler
            now = datetime.now(UTC)

            # Delegate to wrapped handler
            output_events = await self._handler.handle(
                event=payload,
                now=now,
                correlation_id=correlation_id,
            )

            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            logger.info(
                "DispatcherNodeIntrospected processed event",
                extra={
                    "node_id": str(payload.node_id),
                    "output_count": len(output_events),
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.SUCCESS,
                topic="node.introspection",
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                output_count=len(output_events),
                output_events=output_events,
                correlation_id=correlation_id,
            )

        except Exception as e:
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            sanitized_error = sanitize_error_message(e)

            logger.error(
                "DispatcherNodeIntrospected failed: %s",
                sanitized_error,
                extra={
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.HANDLER_ERROR,
                topic="node.introspection",
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
            )

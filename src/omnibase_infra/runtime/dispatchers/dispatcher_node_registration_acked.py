# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dispatcher adapter for HandlerNodeRegistrationAcked.

This module provides a ProtocolMessageDispatcher adapter that wraps
HandlerNodeRegistrationAcked for integration with MessageDispatchEngine.

The adapter:
- Deserializes ModelEventEnvelope payload to ModelNodeRegistrationAcked
- Extracts correlation_id from envelope metadata
- Injects current time via ModelDispatchContext (for ORCHESTRATOR node kind)
- Calls the wrapped handler and emits liveness activation events

Design:
    The adapter follows ONEX dispatcher patterns:
    - Implements ProtocolMessageDispatcher protocol
    - Stateless operation (handler instance is injected)
    - Returns ModelDispatchResult with success/failure status
    - Uses EnumNodeKind.ORCHESTRATOR for time injection

Circuit Breaker Consideration:
    This dispatcher does NOT currently implement MixinAsyncCircuitBreaker because
    it wraps an internal handler (HandlerNodeRegistrationAcked) that performs
    in-process state transitions without external service calls. If the handler
    is modified to make external calls (e.g., database, HTTP, Kafka), consider
    adding circuit breaker protection similar to DispatcherNodeIntrospected.

    See: docs/patterns/dispatcher_resilience.md for implementation guidance.

Related:
    - OMN-888: Registration Orchestrator
    - OMN-892: 2-way Registration E2E Integration Test
"""

from __future__ import annotations

__all__ = ["DispatcherNodeRegistrationAcked"]

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeRegistrationAcked,
    )

logger = logging.getLogger(__name__)


class DispatcherNodeRegistrationAcked:
    """Dispatcher adapter for HandlerNodeRegistrationAcked.

    This dispatcher wraps HandlerNodeRegistrationAcked to integrate it with
    MessageDispatchEngine's category-based routing. It handles:

    - Deserialization: Validates and casts envelope payload to ModelNodeRegistrationAcked
    - Time injection: Uses current time from dispatch context
    - Correlation tracking: Extracts or generates correlation_id
    - Error handling: Returns structured ModelDispatchResult on failure

    Thread Safety:
        This dispatcher is stateless and safe for concurrent invocation.
        The wrapped handler must also be coroutine-safe.

    Attributes:
        _handler: The wrapped HandlerNodeRegistrationAcked instance.

    Example:
        >>> from omnibase_infra.runtime.dispatchers import DispatcherNodeRegistrationAcked
        >>> dispatcher = DispatcherNodeRegistrationAcked(handler_instance)
        >>> result = await dispatcher.handle(envelope)
    """

    def __init__(self, handler: HandlerNodeRegistrationAcked) -> None:
        """Initialize dispatcher with wrapped handler.

        Args:
            handler: HandlerNodeRegistrationAcked instance to delegate to.
        """
        self._handler = handler

    @property
    def dispatcher_id(self) -> str:
        """Unique identifier for this dispatcher.

        Returns:
            str: The dispatcher ID used for registration and tracing.
        """
        return "dispatcher.registration.node-registration-acked"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this dispatcher processes.

        Returns:
            EnumMessageCategory: COMMAND category (ack commands).
        """
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        """Specific message types this dispatcher accepts.

        Returns:
            set[str]: Set containing ModelNodeRegistrationAcked type name.
        """
        return {"ModelNodeRegistrationAcked"}

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
        """Handle registration ack command and return dispatch result.

        Deserializes the envelope payload to ModelNodeRegistrationAcked,
        delegates to the wrapped handler, and returns a structured result.

        Args:
            envelope: Event envelope containing ack command payload.

        Returns:
            ModelDispatchResult: Success with output events or error details.
        """
        started_at = datetime.now(UTC)
        correlation_id = envelope.correlation_id or uuid4()

        try:
            # Validate payload type
            payload = envelope.payload
            if not isinstance(payload, ModelNodeRegistrationAcked):
                # Try to construct from dict if payload is dict-like
                if isinstance(payload, dict):
                    payload = ModelNodeRegistrationAcked.model_validate(payload)
                else:
                    # Reuse started_at timestamp for INVALID_MESSAGE - processing
                    # is minimal (just a type check) so duration is effectively 0
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic="node.registration.acked",
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=started_at,
                        duration_ms=0.0,
                        error_message=f"Expected ModelNodeRegistrationAcked payload, "
                        f"got {type(payload).__name__}",
                        correlation_id=correlation_id,
                        output_events=[],
                    )

            # Assert helps type narrowing after isinstance/model_validate
            assert isinstance(payload, ModelNodeRegistrationAcked)

            # Get current time for handler
            now = datetime.now(UTC)

            # Delegate to wrapped handler
            output_events = await self._handler.handle(
                command=payload,
                now=now,
                correlation_id=correlation_id,
            )

            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            logger.info(
                "DispatcherNodeRegistrationAcked processed command",
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
                topic="node.registration.acked",
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

            logger.exception(
                "DispatcherNodeRegistrationAcked failed: %s",
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
                topic="node.registration.acked",
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
                output_events=[],
            )

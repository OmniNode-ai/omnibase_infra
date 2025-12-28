# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dispatcher adapter for HandlerRuntimeTick.

This module provides a ProtocolMessageDispatcher adapter that wraps
HandlerRuntimeTick for integration with MessageDispatchEngine.

The adapter:
- Deserializes ModelEventEnvelope payload to ModelRuntimeTick
- Extracts correlation_id from envelope metadata
- Injects current time via ModelDispatchContext (for ORCHESTRATOR node kind)
- Calls the wrapped handler and emits timeout events

Design:
    The adapter follows ONEX dispatcher patterns:
    - Implements ProtocolMessageDispatcher protocol
    - Stateless operation (handler instance is injected)
    - Returns ModelDispatchResult with success/failure status
    - Uses EnumNodeKind.ORCHESTRATOR for time injection

Circuit Breaker Consideration:
    This dispatcher does NOT currently implement MixinAsyncCircuitBreaker because
    it wraps an internal handler (HandlerRuntimeTick) that performs in-process
    timeout scanning without external service calls. If the handler is modified
    to make external calls (e.g., database queries, HTTP requests), consider
    adding circuit breaker protection similar to DispatcherNodeIntrospected.

    See: docs/patterns/dispatcher_resilience.md for implementation guidance.

Related:
    - OMN-888: Registration Orchestrator
    - OMN-932: Durable Timeout Handling
    - OMN-892: 2-way Registration E2E Integration Test
"""

from __future__ import annotations

__all__ = ["DispatcherRuntimeTick"]

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerRuntimeTick,
    )

logger = logging.getLogger(__name__)

# Topic identifier used in dispatch results for tracing and observability.
# Note: This is an internal identifier for logging/metrics, NOT the actual Kafka topic name.
# The actual Kafka topic is configured via ModelDispatchRoute.topic_pattern in container_wiring.py.
TOPIC_ID_RUNTIME_TICK = "runtime.tick"


class DispatcherRuntimeTick:
    """Dispatcher adapter for HandlerRuntimeTick.

    This dispatcher wraps HandlerRuntimeTick to integrate it with
    MessageDispatchEngine's category-based routing. It handles:

    - Deserialization: Validates and casts envelope payload to ModelRuntimeTick
    - Time injection: Uses current time from dispatch context
    - Correlation tracking: Extracts or generates correlation_id
    - Error handling: Returns structured ModelDispatchResult on failure

    Thread Safety:
        This dispatcher is stateless and safe for concurrent invocation.
        The wrapped handler must also be coroutine-safe.

    Attributes:
        _handler: The wrapped HandlerRuntimeTick instance.

    Example:
        >>> from omnibase_infra.runtime.dispatchers import DispatcherRuntimeTick
        >>> dispatcher = DispatcherRuntimeTick(handler_instance)
        >>> result = await dispatcher.handle(envelope)
    """

    def __init__(self, handler: HandlerRuntimeTick) -> None:
        """Initialize dispatcher with wrapped handler.

        Args:
            handler: HandlerRuntimeTick instance to delegate to.
        """
        self._handler = handler

    @property
    def dispatcher_id(self) -> str:
        """Unique identifier for this dispatcher.

        Returns:
            str: The dispatcher ID used for registration and tracing.
        """
        return "dispatcher.registration.runtime-tick"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this dispatcher processes.

        Returns:
            EnumMessageCategory: EVENT category (runtime tick events).
        """
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Specific message types this dispatcher accepts.

        Returns:
            set[str]: Set containing ModelRuntimeTick type name.
        """
        return {"ModelRuntimeTick"}

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
        """Handle runtime tick event and return dispatch result.

        Deserializes the envelope payload to ModelRuntimeTick,
        delegates to the wrapped handler, and returns a structured result.

        Args:
            envelope: Event envelope containing runtime tick payload.

        Returns:
            ModelDispatchResult: Success with output events or error details.
        """
        started_at = datetime.now(UTC)
        correlation_id = envelope.correlation_id or uuid4()

        try:
            # Validate payload type
            payload = envelope.payload
            if not isinstance(payload, ModelRuntimeTick):
                # Try to construct from dict if payload is dict-like
                if isinstance(payload, dict):
                    payload = ModelRuntimeTick.model_validate(payload)
                else:
                    completed_at = datetime.now(UTC)
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic=TOPIC_ID_RUNTIME_TICK,
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=(completed_at - started_at).total_seconds() * 1000,
                        error_message=f"Expected ModelRuntimeTick payload, "
                        f"got {type(payload).__name__}",
                        correlation_id=correlation_id,
                    )

            # Assert helps type narrowing after isinstance/model_validate
            assert isinstance(payload, ModelRuntimeTick)

            # Get current time for handler
            now = datetime.now(UTC)

            # Delegate to wrapped handler
            output_events = await self._handler.handle(
                tick=payload,
                now=now,
                correlation_id=correlation_id,
            )

            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            logger.info(
                "DispatcherRuntimeTick processed tick",
                extra={
                    "tick_id": str(payload.tick_id),
                    "output_count": len(output_events),
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.SUCCESS,
                topic=TOPIC_ID_RUNTIME_TICK,
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
                "DispatcherRuntimeTick failed: %s",
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
                topic=TOPIC_ID_RUNTIME_TICK,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
            )

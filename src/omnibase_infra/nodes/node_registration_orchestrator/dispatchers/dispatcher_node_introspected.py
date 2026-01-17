# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: TRY400
# TRY400 disabled: logger.error is intentional to avoid leaking sensitive data in stack traces
"""Dispatcher adapter for HandlerNodeIntrospected.

This module provides a ProtocolMessageDispatcher adapter that wraps
HandlerNodeIntrospected for integration with MessageDispatchEngine.

The adapter:
- Deserializes ModelEventEnvelope payload to ModelNodeIntrospectionEvent
- Extracts correlation_id from envelope metadata
- Injects current time via ModelDispatchContext (for ORCHESTRATOR node kind)
- Calls the wrapped handler and emits output events
- Provides circuit breaker resilience via MixinAsyncCircuitBreaker

Design:
    The adapter follows ONEX dispatcher patterns:
    - Implements ProtocolMessageDispatcher protocol
    - Uses MixinAsyncCircuitBreaker for fault tolerance
    - Stateless operation (handler instance is injected)
    - Returns ModelDispatchResult with success/failure status
    - Uses EnumNodeKind.ORCHESTRATOR for time injection

Circuit Breaker Pattern:
    - Uses MixinAsyncCircuitBreaker for resilience against handler failures
    - Configured for KAFKA transport (threshold=3, reset_timeout=20.0s)
    - Opens circuit after 3 consecutive failures to prevent cascading issues
    - Transitions to HALF_OPEN after timeout to test recovery
    - Raises InfraUnavailableError when circuit is OPEN

Typing Note (ModelEventEnvelope[object]):
    The ``handle()`` method uses ``ModelEventEnvelope[object]`` instead of ``Any``
    per CLAUDE.md guidance: "Use ``object`` for generic payloads".

    This is intentional:
    - CLAUDE.md mandates "NEVER use ``Any``" for type annotations
    - Generic dispatchers must accept envelopes with any payload type at the
      protocol level (routing is based on topic/category/message_type)
    - Payload extraction uses ``isinstance()`` type guards for runtime safety::

        payload = envelope.payload
        if not isinstance(payload, ModelNodeIntrospectionEvent):
            # Attempt deserialization from dict
            ...

    - ``object`` provides better type safety than ``Any`` while allowing the
      flexibility required for polymorphic dispatch

Related:
    - OMN-888: Registration Orchestrator
    - OMN-892: 2-way Registration E2E Integration Test
    - OMN-1346: Registration Code Extraction
    - docs/patterns/dispatcher_resilience.md
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumDispatchStatus,
    EnumInfraTransportType,
    EnumMessageCategory,
)
from omnibase_infra.errors import (
    EnvelopeValidationError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
    )

__all__ = ["DispatcherNodeIntrospected"]

logger = logging.getLogger(__name__)

# Topic identifier used in dispatch results for tracing and observability.
# Note: Internal identifier for logging/metrics, NOT the actual Kafka topic.
# Actual topic is configured via ModelDispatchRoute.topic_pattern.
TOPIC_ID_NODE_INTROSPECTION = "node.introspection"


class DispatcherNodeIntrospected(MixinAsyncCircuitBreaker):
    """Dispatcher adapter for HandlerNodeIntrospected.

    This dispatcher wraps HandlerNodeIntrospected to integrate it with
    MessageDispatchEngine's category-based routing. It handles:

    - Deserialization: Validates and casts payload to ModelNodeIntrospectionEvent
    - Time injection: Uses current time from dispatch context
    - Correlation tracking: Extracts or generates correlation_id
    - Error handling: Returns structured ModelDispatchResult on failure
    - Circuit breaker: Fault tolerance via MixinAsyncCircuitBreaker

    Circuit Breaker Configuration:
        - threshold: 3 consecutive failures before opening circuit
        - reset_timeout: 20.0 seconds before attempting recovery
        - transport_type: KAFKA (event dispatching transport)
        - service_name: dispatcher.registration.node-introspected

    Thread Safety:
        This dispatcher uses asyncio.Lock for coroutine-safe circuit breaker
        state management. The wrapped handler must also be coroutine-safe.

    Attributes:
        _handler: The wrapped HandlerNodeIntrospected instance.

    Example:
        >>> from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
        ...     DispatcherNodeIntrospected,
        ... )
        >>> dispatcher = DispatcherNodeIntrospected(handler_instance)
        >>> result = await dispatcher.handle(envelope)
    """

    def __init__(self, handler: HandlerNodeIntrospected) -> None:
        """Initialize dispatcher with wrapped handler and circuit breaker.

        Args:
            handler: HandlerNodeIntrospected instance to delegate to.

        Circuit Breaker:
            Initialized with KAFKA transport settings per dispatcher_resilience.md:
            - threshold=3: Open after 3 consecutive failures
            - reset_timeout=20.0: 20 seconds before testing recovery
        """
        self._handler = handler

        # Initialize circuit breaker using mixin pattern
        # Configuration follows docs/patterns/dispatcher_resilience.md guidelines
        self._init_circuit_breaker(
            threshold=3,  # Open after 3 failures (KAFKA is critical)
            reset_timeout=20.0,  # 20 seconds recovery window
            service_name="dispatcher.registration.node-introspected",
            transport_type=EnumInfraTransportType.KAFKA,
        )

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

        Circuit Breaker Integration:
            - Checks circuit state before processing (raises if OPEN)
            - Records failures to track service health
            - Resets on success to maintain circuit health
            - InfraUnavailableError propagates to caller for DLQ handling

        Args:
            envelope: Event envelope containing introspection payload.

        Returns:
            ModelDispatchResult: Success with output events or error details.

        Raises:
            InfraUnavailableError: If circuit breaker is OPEN.
        """
        started_at = datetime.now(UTC)
        correlation_id = envelope.correlation_id or uuid4()

        # Check circuit breaker before processing (coroutine-safe)
        # If circuit is OPEN, raises InfraUnavailableError immediately
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("handle", correlation_id)

        try:
            # Validate payload type
            payload = envelope.payload
            if not isinstance(payload, ModelNodeIntrospectionEvent):
                # Try to construct from dict if payload is dict-like
                if isinstance(payload, dict):
                    payload = ModelNodeIntrospectionEvent.model_validate(payload)
                else:
                    # Reuse started_at timestamp for INVALID_MESSAGE - processing
                    # is minimal (just a type check) so duration is effectively 0
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic=TOPIC_ID_NODE_INTROSPECTION,
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=started_at,
                        duration_ms=0.0,
                        error_message=f"Expected ModelNodeIntrospectionEvent payload, "
                        f"got {type(payload).__name__}",
                        correlation_id=correlation_id,
                        output_events=[],
                    )

            # Explicit type guard (not assert) for production safety
            # Type narrowing after isinstance/model_validate above
            if not isinstance(payload, ModelNodeIntrospectionEvent):
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="handle_introspection",
                    correlation_id=correlation_id,
                )
                raise EnvelopeValidationError(
                    f"Expected ModelNodeIntrospectionEvent after validation, "
                    f"got {type(payload).__name__}",
                    context=context,
                )

            # Get current time for handler
            now = datetime.now(UTC)

            # Create envelope for handler (ProtocolMessageHandler signature)
            handler_envelope: ModelEventEnvelope[ModelNodeIntrospectionEvent] = (
                ModelEventEnvelope(
                    envelope_id=uuid4(),
                    payload=payload,
                    envelope_timestamp=now,
                    correlation_id=correlation_id,
                    source=self.dispatcher_id,
                )
            )

            # Delegate to wrapped handler
            handler_output = await self._handler.handle(handler_envelope)
            output_events = list(handler_output.events)

            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            # Record success for circuit breaker (coroutine-safe)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

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
                topic=TOPIC_ID_NODE_INTROSPECTION,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                output_count=len(output_events),
                output_events=output_events,
                correlation_id=correlation_id,
            )

        except ValidationError as e:
            # ValidationError indicates malformed message payload - not a handler error
            # Return INVALID_MESSAGE to route to DLQ without retry
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            sanitized_error = sanitize_error_message(e)

            logger.warning(
                "DispatcherNodeIntrospected received invalid message: %s",
                sanitized_error,
                extra={
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                    "error_type": "ValidationError",
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.INVALID_MESSAGE,
                topic=TOPIC_ID_NODE_INTROSPECTION,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
                output_events=[],
            )

        except InfraUnavailableError:
            # Circuit breaker errors should propagate for engine-level handling
            # (e.g., routing to DLQ)
            raise

        except Exception as e:
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            sanitized_error = sanitize_error_message(e)

            # Record failure for circuit breaker (coroutine-safe)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("handle", correlation_id)

            # Use logger.error instead of logger.exception to avoid leaking
            # potentially sensitive data in stack traces (credentials, PII, etc.)
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
                topic=TOPIC_ID_NODE_INTROSPECTION,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
                output_events=[],
            )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: TRY400
# TRY400 disabled: logger.error is intentional to avoid leaking sensitive data in stack traces
"""Dispatcher adapter for HandlerNodeIntrospected.

ProtocolMessageDispatcher adapter that wraps HandlerNodeIntrospected for
integration with MessageDispatchEngine.

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
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
from omnibase_infra.enums import (
    EnumDispatchStatus,
    EnumInfraTransportType,
    EnumMessageCategory,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.events.model_node_registration_accepted import (
    ModelNodeRegistrationAccepted,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers._util_envelope_extract import (
    extract_envelope_fields,
)
from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION_ACKED
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

_ENV_AUTO_ACK = "ONEX_REGISTRATION_AUTO_ACK"
_ACK_TOPIC: str = SUFFIX_NODE_REGISTRATION_ACKED


def _auto_ack_enabled() -> bool:
    """Return True when ONEX_REGISTRATION_AUTO_ACK=true in the environment.

    Intended for local/dev use only — in production external nodes send their
    own ack command, bypassing the distributed handshake.  See OMN-3444.
    """
    return os.environ.get(_ENV_AUTO_ACK, "false").lower() == "true"


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

    def __init__(
        self,
        handler: HandlerNodeIntrospected,
        event_bus: ProtocolEventBus | None = None,
    ) -> None:
        """Initialize dispatcher with wrapped handler and circuit breaker.

        Args:
            handler: HandlerNodeIntrospected instance to delegate to.
            event_bus: Optional event bus for direct-publishing the auto-ACK
                command (Path B, OMN-3444). When None, auto-ACK is silently
                skipped even if ONEX_REGISTRATION_AUTO_ACK=true.

        Circuit Breaker:
            Initialized with KAFKA transport settings per dispatcher_resilience.md:
            - threshold=3: Open after 3 consecutive failures
            - reset_timeout=20.0: 20 seconds before testing recovery
        """
        self._handler = handler
        self._event_bus = event_bus

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
            set[str]: Set containing both Python class name and ONEX event_type
                routing key for backwards compatibility and routing flexibility.
        """
        return {"ModelNodeIntrospectionEvent", "platform.node-introspection"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """ONEX node kind for time injection rules.

        Returns:
            EnumNodeKind: ORCHESTRATOR for workflow coordination with time.
        """
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self,
        envelope: ModelEventEnvelope[object] | dict[str, object],
    ) -> ModelDispatchResult:
        """Handle introspection event and return dispatch result.

        Deserializes the envelope payload to ModelNodeIntrospectionEvent,
        delegates to the wrapped handler, and returns a structured result.

        The dispatch engine materializes envelopes to dicts before calling
        dispatchers (serialization boundary). This method accepts both
        ModelEventEnvelope objects and materialized dicts.

        Circuit Breaker Integration:
            - Checks circuit state before processing (raises if OPEN)
            - Records failures to track service health
            - Resets on success to maintain circuit health
            - InfraUnavailableError propagates to caller for DLQ handling

        Args:
            envelope: Event envelope or materialized dict from dispatch engine.
                Dict format: {"payload": {...}, "__bindings": {...}, "__debug_trace": {...}}

        Returns:
            ModelDispatchResult: Success with output events or error details.

        Raises:
            InfraUnavailableError: If circuit breaker is OPEN.
        """
        # NOTE: Both started_at and handler 'now' use direct datetime.now(UTC)
        # instead of ModelDispatchContext.now due to protocol signature limitation.
        # See TODO(OMN-2050) below for details.
        started_at = datetime.now(UTC)

        correlation_id, raw_payload = extract_envelope_fields(envelope)

        # Check circuit breaker before processing (coroutine-safe)
        # If circuit is OPEN, raises InfraUnavailableError immediately
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("handle", correlation_id)

        try:
            # Validate payload type
            payload = raw_payload
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

            # Type narrowing: the branch above guarantees payload is
            # ModelNodeIntrospectionEvent (isinstance returned True, or model_validate
            # succeeded, or we returned early). Use cast() instead of a redundant
            # runtime isinstance check to satisfy mypy.
            payload = cast("ModelNodeIntrospectionEvent", payload)

            # TODO(OMN-2050): Use injected time from ModelDispatchContext instead
            # of datetime.now(UTC). Currently, the ProtocolMessageDispatcher.handle()
            # signature accepts only the envelope, so there is no way to receive the
            # dispatch engine's ModelDispatchContext.now timestamp. When the protocol
            # is updated to pass ModelDispatchContext (or the envelope carries a
            # dispatch_timestamp field), replace this direct clock access with the
            # injected value for full time-injection compliance.
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
            output_intents = handler_output.intents

            # Auto-ACK (Path B, OMN-3444): direct-publish ack when the reducer transitions
            # to AWAITING_ACK. Gate: ModelNodeRegistrationAccepted in output events.
            # Published directly to _ACK_TOPIC as a command (not via output_events).
            # This is intentional: ACK commands are not in published_events (they are commands,
            # not events), so they are not routed by the topic_router. Direct publish is correct here.
            # NOTE: The "wrong topic" routing bug (OMN-4880) has been fixed — output_events are
            # now routed per-event-type via DispatchResultApplier.topic_router.
            if (
                _auto_ack_enabled()
                and self._event_bus is not None
                and any(
                    isinstance(e, ModelNodeRegistrationAccepted) for e in output_events
                )
            ):
                auto_ack_payload = ModelNodeRegistrationAcked(
                    node_id=payload.node_id,
                    correlation_id=correlation_id,
                    timestamp=now,
                )
                ack_envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                    envelope_id=uuid4(),
                    payload=auto_ack_payload,
                    envelope_timestamp=now,
                    correlation_id=correlation_id,
                    source=self.dispatcher_id,
                )
                # ModelEventEnvelope is structurally compatible with ProtocolEventEnvelope
                # but lacks the async get_payload() method; mixin_node_introspection uses
                # the same pattern (mixin_node_introspection.py:2276).
                await self._event_bus.publish_envelope(
                    ack_envelope,  # type: ignore[arg-type]
                    topic=_ACK_TOPIC,
                )
                logger.debug(
                    "Auto-ACK published for node %s (ONEX_REGISTRATION_AUTO_ACK=true)",
                    payload.node_id,
                    extra={
                        "node_id": str(payload.node_id),
                        "correlation_id": str(correlation_id),
                    },
                )

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
                output_intents=output_intents,
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

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
# ruff: noqa: TRY400
# TRY400 disabled: logger.error is intentional to avoid leaking sensitive data in stack traces
"""Dispatcher adapter for HandlerTopicCatalogQuery.

This module provides a ``ProtocolMessageDispatcher`` adapter that wraps
``HandlerTopicCatalogQuery`` for integration with ``MessageDispatchEngine``.

The adapter:
- Deserializes ``ModelEventEnvelope`` payload to ``ModelTopicCatalogQuery``
- Extracts ``correlation_id`` from envelope metadata
- Calls the wrapped handler and collects output events
- Provides circuit breaker resilience via ``MixinAsyncCircuitBreaker``
- Returns ``ModelDispatchResult`` with success/failure status

Design:
    Follows the ONEX dispatcher pattern established by
    ``dispatcher_node_introspected.py``:
    - Implements ``ProtocolMessageDispatcher`` protocol
    - Uses ``MixinAsyncCircuitBreaker`` for fault tolerance
    - Stateless operation (handler instance is injected)
    - Returns ``ModelDispatchResult`` with success/failure status
    - Message category: COMMAND (query topics are command semantics)

Circuit Breaker Pattern:
    - Configured for KAFKA transport (threshold=3, reset_timeout=20.0s)
    - Opens circuit after 3 consecutive failures
    - Transitions to HALF_OPEN after timeout to test recovery
    - Raises ``InfraUnavailableError`` when circuit is OPEN

Typing Note (ModelEventEnvelope[object]):
    Uses ``object`` instead of ``Any`` per CLAUDE.md guidance. Payload
    extraction uses ``isinstance()`` type guards for runtime safety.

Related:
    - OMN-2313: Topic Catalog: query handler + dispatcher + contract wiring
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching
    - docs/patterns/dispatcher_resilience.md
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumDispatchStatus,
    EnumInfraTransportType,
    EnumMessageCategory,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers._util_envelope_extract import (
    extract_envelope_fields,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_topic_catalog_query import (
        HandlerTopicCatalogQuery,
    )

__all__ = ["DispatcherTopicCatalogQuery"]

logger = logging.getLogger(__name__)

# Topic identifier used in dispatch results for tracing and observability.
# Note: Internal identifier for logging/metrics, NOT the actual Kafka topic.
TOPIC_ID_TOPIC_CATALOG_QUERY = "platform.topic-catalog-query"


class DispatcherTopicCatalogQuery(MixinAsyncCircuitBreaker):
    """Dispatcher adapter for HandlerTopicCatalogQuery.

    Wraps ``HandlerTopicCatalogQuery`` to integrate it with
    ``MessageDispatchEngine``'s category-based routing. Handles:

    - Deserialization: Validates and casts payload to ``ModelTopicCatalogQuery``
    - Correlation tracking: Extracts or generates ``correlation_id``
    - Error handling: Returns structured ``ModelDispatchResult`` on failure
    - Circuit breaker: Fault tolerance via ``MixinAsyncCircuitBreaker``

    Circuit Breaker Configuration:
        - threshold: 3 consecutive failures before opening circuit
        - reset_timeout: 20.0 seconds before attempting recovery
        - transport_type: KAFKA (event dispatching transport)
        - service_name: dispatcher.registration.topic-catalog-query

    Thread Safety:
        Uses ``asyncio.Lock`` for coroutine-safe circuit breaker state
        management. The wrapped handler must also be coroutine-safe.

    Attributes:
        _handler: The wrapped ``HandlerTopicCatalogQuery`` instance.

    Example:
        >>> dispatcher = DispatcherTopicCatalogQuery(handler_instance)
        >>> result = await dispatcher.handle(envelope)
    """

    def __init__(self, handler: HandlerTopicCatalogQuery) -> None:
        """Initialize dispatcher with wrapped handler and circuit breaker.

        Args:
            handler: ``HandlerTopicCatalogQuery`` instance to delegate to.

        Circuit Breaker:
            Initialized with KAFKA transport settings per
            ``docs/patterns/dispatcher_resilience.md``:
            - threshold=3: Open after 3 consecutive failures
            - reset_timeout=20.0: 20 seconds before testing recovery
        """
        self._handler = handler

        self._init_circuit_breaker(
            threshold=3,
            reset_timeout=20.0,
            service_name="dispatcher.registration.topic-catalog-query",
            transport_type=EnumInfraTransportType.KAFKA,
        )

    @property
    def dispatcher_id(self) -> str:
        """Unique identifier for this dispatcher."""
        return "dispatcher.registration.topic-catalog-query"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this dispatcher processes.

        COMMAND category: catalog queries follow request-response semantics
        where the querier sends a command expecting a response.
        """
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        """Specific message types this dispatcher accepts."""
        return {"ModelTopicCatalogQuery", "platform.topic-catalog-query"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """ONEX node kind for time injection rules."""
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self,
        envelope: ModelEventEnvelope[object] | dict[str, object],
    ) -> ModelDispatchResult:
        """Handle topic catalog query and return dispatch result.

        Deserializes the envelope payload to ``ModelTopicCatalogQuery``,
        delegates to the wrapped handler, and returns a structured result.

        Circuit Breaker Integration:
            - Checks circuit state before processing (raises if OPEN)
            - Records failures to track service health
            - Resets on success to maintain circuit health
            - ``InfraUnavailableError`` propagates to caller for DLQ handling

        Args:
            envelope: Event envelope or materialized dict from dispatch engine.

        Returns:
            ``ModelDispatchResult``: Success with output events or error details.

        Raises:
            ``InfraUnavailableError``: If circuit breaker is OPEN.
        """
        started_at = datetime.now(UTC)

        correlation_id, raw_payload = extract_envelope_fields(envelope)

        # Check circuit breaker before processing (coroutine-safe)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("handle", correlation_id)

        try:
            # Validate payload type
            payload = raw_payload
            if not isinstance(payload, ModelTopicCatalogQuery):
                if isinstance(payload, dict):
                    payload = ModelTopicCatalogQuery.model_validate(payload)
                else:
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic=TOPIC_ID_TOPIC_CATALOG_QUERY,
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=started_at,
                        duration_ms=0.0,
                        error_message=(
                            f"Expected ModelTopicCatalogQuery payload, "
                            f"got {type(payload).__name__}"
                        ),
                        correlation_id=correlation_id,
                        output_events=[],
                    )

            payload = cast("ModelTopicCatalogQuery", payload)

            now = datetime.now(UTC)

            handler_envelope: ModelEventEnvelope[ModelTopicCatalogQuery] = (
                ModelEventEnvelope(
                    envelope_id=uuid4(),
                    payload=payload,
                    envelope_timestamp=now,
                    correlation_id=correlation_id,
                    source=self.dispatcher_id,
                )
            )

            handler_output = await self._handler.handle(handler_envelope)
            output_events = list(handler_output.events)

            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            # Record success for circuit breaker (coroutine-safe)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.info(
                "DispatcherTopicCatalogQuery processed query",
                extra={
                    "client_id": payload.client_id,
                    "output_count": len(output_events),
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.SUCCESS,
                topic=TOPIC_ID_TOPIC_CATALOG_QUERY,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                output_count=len(output_events),
                output_events=output_events,
                correlation_id=correlation_id,
            )

        except ValidationError as e:
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            sanitized_error = sanitize_error_message(e)

            logger.warning(
                "DispatcherTopicCatalogQuery received invalid message: %s",
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
                topic=TOPIC_ID_TOPIC_CATALOG_QUERY,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
                output_events=[],
            )

        except InfraUnavailableError:
            # Circuit breaker errors propagate for engine-level handling
            raise

        except Exception as e:
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            sanitized_error = sanitize_error_message(e)

            # Record failure for circuit breaker (coroutine-safe)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("handle", correlation_id)

            logger.error(
                "DispatcherTopicCatalogQuery failed: %s",
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
                topic=TOPIC_ID_TOPIC_CATALOG_QUERY,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
                output_events=[],
            )

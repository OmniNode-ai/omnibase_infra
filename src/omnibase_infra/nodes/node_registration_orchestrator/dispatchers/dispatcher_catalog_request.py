# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
# ruff: noqa: TRY400
# TRY400 disabled: logger.error is intentional to avoid leaking sensitive data in stack traces
"""Dispatcher adapter for HandlerCatalogRequest.

This module provides a ``ProtocolMessageDispatcher`` adapter that wraps
``HandlerCatalogRequest`` for integration with ``MessageDispatchEngine``.

The adapter:
- Deserializes ``ModelEventEnvelope`` payload to ``ModelTopicCatalogRequest``
- Extracts ``correlation_id`` from envelope or payload
- Calls the wrapped handler and collects output events
- Provides circuit breaker resilience via ``MixinAsyncCircuitBreaker``
- Returns ``ModelDispatchResult`` with success/failure status

Design:
    Follows the ONEX dispatcher pattern established by
    ``dispatcher_topic_catalog_query.py``:
    - Implements ``ProtocolMessageDispatcher`` protocol
    - Uses ``MixinAsyncCircuitBreaker`` for fault tolerance
    - Stateless operation (handler instance is injected)
    - Returns ``ModelDispatchResult`` with success/failure status
    - Message category: COMMAND (catalog requests are command semantics)

Circuit Breaker Pattern:
    - Configured for KAFKA transport (threshold=3, reset_timeout=20.0s)
    - Opens circuit after 3 consecutive failures
    - Transitions to HALF_OPEN after timeout to test recovery
    - Raises ``InfraUnavailableError`` when circuit is OPEN

Related:
    - OMN-2923: Catalog responder for topic-catalog-request.v1
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
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.registration.model_topic_catalog_request import (
    ModelTopicCatalogRequest,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers._util_envelope_extract import (
    extract_envelope_fields,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_catalog_request import (
        HandlerCatalogRequest,
    )

__all__ = ["DispatcherCatalogRequest"]

logger = logging.getLogger(__name__)

# Topic identifier used in dispatch results for tracing and observability.
# Note: Internal identifier for logging/metrics, NOT the actual Kafka topic.
TOPIC_ID_CATALOG_REQUEST = "platform.request-introspection"


class DispatcherCatalogRequest(MixinAsyncCircuitBreaker):
    """Dispatcher adapter for HandlerCatalogRequest.

    Wraps ``HandlerCatalogRequest`` to integrate it with
    ``MessageDispatchEngine``'s category-based routing. Handles:

    - Deserialization: Validates and casts payload to ``ModelTopicCatalogRequest``
    - Correlation tracking: Extracts or generates ``correlation_id``
    - Error handling: Returns structured ``ModelDispatchResult`` on failure
    - Circuit breaker: Fault tolerance via ``MixinAsyncCircuitBreaker``

    Circuit Breaker Configuration:
        - threshold: 3 consecutive failures before opening circuit
        - reset_timeout: 20.0 seconds before attempting recovery
        - transport_type: KAFKA (event dispatching transport)
        - service_name: dispatcher.registration.catalog-request

    Attributes:
        _handler: The wrapped ``HandlerCatalogRequest`` instance.
    """

    def __init__(self, handler: HandlerCatalogRequest) -> None:
        """Initialize dispatcher with wrapped handler and circuit breaker.

        Args:
            handler: ``HandlerCatalogRequest`` instance to delegate to.
        """
        self._handler = handler

        self._init_circuit_breaker(
            threshold=3,
            reset_timeout=20.0,
            service_name="dispatcher.registration.catalog-request",
            transport_type=EnumInfraTransportType.KAFKA,
        )

    @property
    def dispatcher_id(self) -> str:
        """Unique identifier for this dispatcher."""
        return "dispatcher.registration.catalog-request"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this dispatcher processes.

        COMMAND category: catalog requests follow request-response semantics.
        """
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        """Specific message types this dispatcher accepts."""
        return {"ModelTopicCatalogRequest", "platform.request-introspection"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """ONEX node kind for time injection rules."""
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self,
        envelope: ModelEventEnvelope[object] | dict[str, object],
    ) -> ModelDispatchResult:
        """Handle a catalog request and return dispatch result.

        Deserializes the envelope payload to ``ModelTopicCatalogRequest``,
        delegates to the wrapped handler, and returns a structured result.

        Circuit Breaker Integration:
            - Checks circuit state before processing (raises if OPEN)
            - Records failures to track service health
            - Resets on success to maintain circuit health

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
            if not isinstance(payload, ModelTopicCatalogRequest):
                if isinstance(payload, dict):
                    payload = ModelTopicCatalogRequest.model_validate(payload)
                else:
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic=TOPIC_ID_CATALOG_REQUEST,
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=started_at,
                        duration_ms=0.0,
                        error_message=(
                            f"Expected ModelTopicCatalogRequest payload, "
                            f"got {type(payload).__name__}"
                        ),
                        correlation_id=correlation_id,
                        output_events=[],
                    )

            now = datetime.now(UTC)

            handler_envelope: ModelEventEnvelope[ModelTopicCatalogRequest] = (
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

            # Count topics in the response for observability
            topic_count = 0
            if output_events and isinstance(
                output_events[0], ModelTopicCatalogResponse
            ):
                topic_count = len(output_events[0].topics)

            logger.info(
                "DispatcherCatalogRequest processed request",
                extra={
                    "topic_count": topic_count,
                    "output_count": len(output_events),
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.SUCCESS,
                topic=TOPIC_ID_CATALOG_REQUEST,
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
                "DispatcherCatalogRequest received invalid message: %s",
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
                topic=TOPIC_ID_CATALOG_REQUEST,
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
                "DispatcherCatalogRequest failed: %s",
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
                topic=TOPIC_ID_CATALOG_REQUEST,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitized_error,
                correlation_id=correlation_id,
                output_events=[],
            )

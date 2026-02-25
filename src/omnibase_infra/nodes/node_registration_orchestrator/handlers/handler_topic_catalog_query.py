# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for ModelTopicCatalogQuery - topic catalog request-response.

This handler processes topic catalog query requests published by dashboard
or other consumers on the ``topic-catalog-query`` topic. It delegates to
``ServiceTopicCatalog.build_catalog()`` and returns a
``ModelTopicCatalogResponse`` event.

Architecture:
    Follows the same stateless, coroutine-safe pattern as
    ``handler_node_introspected.py``:

    1. Extract payload from envelope
    2. Delegate I/O to injected ``ServiceTopicCatalog``
    3. Return ``ModelHandlerOutput`` with a single ``ModelTopicCatalogResponse``
       event regardless of success or partial failure

    The handler NEVER fails silently. All error conditions are surfaced via
    the ``warnings`` field of ``ModelTopicCatalogResponse``.

Error Handling:
    - Malformed query payload: log warning, return empty response with
      ``warnings=(INVALID_QUERY_PAYLOAD,)``
    - Consul unavailable: ``ServiceTopicCatalog`` already returns partial
      success with ``warnings=(CONSUL_UNAVAILABLE,)`` or
      ``warnings=(CONSUL_SCAN_TIMEOUT,)``
    - Unexpected exception: log error, return empty response with
      ``warnings=(INTERNAL_ERROR,)``

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different event instances.

Related Tickets:
    - OMN-2313: Topic Catalog: query handler + dispatcher + contract wiring
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching
    - OMN-2310: Topic Catalog model + suffix foundation
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.models.catalog.catalog_warning_codes import (
    INTERNAL_ERROR,
    INVALID_QUERY_PAYLOAD,
)
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.services.protocol_topic_catalog_service import (
    ProtocolTopicCatalogService,
)

logger = logging.getLogger(__name__)


class HandlerTopicCatalogQuery:
    """Handler for ModelTopicCatalogQuery - topic catalog request-response.

    Processes catalog query requests and returns a full catalog snapshot via
    ``ModelTopicCatalogResponse``. This handler is the primary consumer of
    ``ServiceTopicCatalog.build_catalog()``.

    Dependency Injection:
        Receives ``ServiceTopicCatalog`` via constructor injection. The service
        is wired through the registry's ``handler_dependencies`` pattern in
        ``RegistryInfraNodeRegistrationOrchestrator.create_registry()``.

    Always-Respond Contract:
        This handler ALWAYS returns a ``ModelHandlerOutput`` with exactly one
        event (``ModelTopicCatalogResponse``). It never raises exceptions to
        the caller - error conditions are encoded in the ``warnings`` field of
        the response.

    Idempotency:
        Same ``correlation_id`` on the same catalog state yields equivalent
        responses (modulo timestamp differences).

    Attributes:
        _catalog_service: ServiceTopicCatalog for building catalog snapshots.

    Example:
        >>> service = ServiceTopicCatalogPostgres(container=container, pool=pool)
        >>> handler = HandlerTopicCatalogQuery(catalog_service=service)
        >>> output = await handler.handle(envelope)
        >>> assert len(output.events) == 1
        >>> response = output.events[0]
        >>> isinstance(response, ModelTopicCatalogResponse)
        True
    """

    def __init__(
        self,
        catalog_service: ProtocolTopicCatalogService,
    ) -> None:
        """Initialize the handler with a topic catalog service.

        Args:
            catalog_service: Any implementation of ``ProtocolTopicCatalogService``:
                either ``ServiceTopicCatalog`` (Consul-backed, legacy) or
                ``ServiceTopicCatalogPostgres`` (PostgreSQL-backed, OMN-2746).
                Wired via registry ``handler_dependencies``.
        """
        self._catalog_service = catalog_service

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-topic-catalog-query"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelTopicCatalogQuery"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler.

        Returns INFRA_HANDLER because this handler coordinates platform-level
        infrastructure queries (topic catalog discovery).
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns EFFECT because this handler performs I/O via the catalog
        service (reads from PostgreSQL or Consul KV depending on the
        injected implementation).
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelTopicCatalogQuery],
    ) -> ModelHandlerOutput[object]:
        """Process a topic catalog query and return a catalog response.

        Delegates to ``ServiceTopicCatalog.build_catalog()`` and wraps the
        result in a ``ModelHandlerOutput`` with a single event. Never raises
        to the caller - all error conditions are encoded in the response
        ``warnings`` field.

        Args:
            envelope: Event envelope containing ``ModelTopicCatalogQuery`` payload.

        Returns:
            ``ModelHandlerOutput`` with exactly one event:
            ``ModelTopicCatalogResponse``. The response may contain warnings
            for partial-success scenarios (Consul unavailable, timeout, etc.).
        """
        start_time = time.perf_counter()
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Validate payload - always respond, never fail silently
        payload = envelope.payload
        if not isinstance(payload, ModelTopicCatalogQuery):
            logger.warning(
                "HandlerTopicCatalogQuery received unexpected payload type: %s",
                type(payload).__name__,
                extra={"correlation_id": str(correlation_id)},
            )
            empty_response = self._empty_response(
                correlation_id=correlation_id,
                warnings=(INVALID_QUERY_PAYLOAD,),
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            return ModelHandlerOutput(
                input_envelope_id=envelope.envelope_id,
                correlation_id=correlation_id,
                handler_id=self.handler_id,
                node_kind=self.node_kind,
                events=(empty_response,),
                intents=(),
                projections=(),
                result=None,
                processing_time_ms=processing_time_ms,
                timestamp=now,
            )

        logger.debug(
            "HandlerTopicCatalogQuery processing query from client=%s",
            payload.client_id,
            extra={
                "correlation_id": str(correlation_id),
                "client_id": payload.client_id,
                "include_inactive": payload.include_inactive,
                "topic_pattern": payload.topic_pattern,
            },
        )

        # Delegate to ServiceTopicCatalog - it handles all partial-success logic
        try:
            catalog_response = await self._catalog_service.build_catalog(
                correlation_id=correlation_id,
                include_inactive=payload.include_inactive,
                topic_pattern=payload.topic_pattern,
            )
        except Exception as e:
            logger.error(  # noqa: TRY400
                "HandlerTopicCatalogQuery: unexpected error from ServiceTopicCatalog: %s",
                type(e).__name__,
                extra={"correlation_id": str(correlation_id)},
            )
            catalog_response = self._empty_response(
                correlation_id=correlation_id,
                warnings=(INTERNAL_ERROR,),
            )

        if catalog_response.warnings:
            logger.info(
                "HandlerTopicCatalogQuery returning response with %d warnings: %s",
                len(catalog_response.warnings),
                catalog_response.warnings,
                extra={
                    "correlation_id": str(correlation_id),
                    "topic_count": len(catalog_response.topics),
                },
            )
        else:
            logger.info(
                "HandlerTopicCatalogQuery returning %d topics",
                len(catalog_response.topics),
                extra={
                    "correlation_id": str(correlation_id),
                    "catalog_version": catalog_response.catalog_version,
                },
            )

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=(catalog_response,),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )

    def _empty_response(
        self,
        correlation_id: UUID,
        warnings: tuple[str, ...],
    ) -> ModelTopicCatalogResponse:
        """Build an empty catalog response for error conditions.

        Args:
            correlation_id: Correlation ID to embed in the response.
            warnings: Warning tokens describing the error condition.

        Returns:
            ``ModelTopicCatalogResponse`` with zero topics and the given warnings.
        """
        # catalog_version=0 is a sentinel value for error-condition responses only.
        # A real catalog always has version >= 1 (set by ServiceTopicCatalog).
        # Callers that need to distinguish error responses from a legitimately
        # empty catalog should check ``warnings`` rather than ``catalog_version``.
        return ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=(),
            catalog_version=0,
            node_count=0,
            generated_at=datetime.now(UTC),
            warnings=warnings,
        )


__all__: list[str] = ["HandlerTopicCatalogQuery"]

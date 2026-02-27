# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for ModelTopicCatalogRequest - introspection-based catalog responder.

This handler processes catalog requests published on
``onex.cmd.platform.request-introspection.v1``. It reads the orchestrator's
accumulated introspection state (via ``ServiceIntrospectionTopicStore``) and
returns a ``ModelTopicCatalogResponse`` containing all registered ``onex.evt.*``
topics.

Architecture:
    Follows the stateless handler pattern from handler_topic_catalog_query.py:

    1. Extract correlation_id from envelope or payload
    2. Query ServiceIntrospectionTopicStore for accumulated evt topics
    3. Return ModelHandlerOutput with a single ModelTopicCatalogResponse event

    The handler NEVER fails silently. All error conditions are surfaced via
    the ``warnings`` field of the catalog response.

Cold-Start Behavior:
    If no introspection state has accumulated yet (no nodes have registered),
    the handler responds immediately with:
    ``topics=(), warnings=["Registry cold — no nodes registered yet"]``

    It NEVER holds the response or times out.

Topic Filter Rule:
    Only topics starting with ``onex.evt.`` are included in the response.
    The filter is applied as a simple string prefix check, not regex.

Wire Format:
    The response is serialized as JSON and published on
    ``onex.evt.platform.topic-catalog-response.v1``. The omnidash
    ``TopicCatalogManager`` parses responses with the schema:

    ```json
    {
        "correlation_id": "...",
        "topics": [{"topic_name": "onex.evt.foo.bar.v1"}, ...],
        "warnings": []
    }
    ```

    This handler uses the existing ``ModelTopicCatalogResponse`` model with
    ``ModelTopicCatalogEntry`` entries to match this wire format.

Coroutine Safety:
    This handler is stateless with respect to its own fields and is
    coroutine-safe for concurrent calls with different event instances.
    The ``ServiceIntrospectionTopicStore`` uses asyncio.Lock internally.

Related Tickets:
    - OMN-2923: Catalog responder for topic-catalog-request.v1
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
from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.models.registration.model_topic_catalog_request import (
    ModelTopicCatalogRequest,
)
from omnibase_infra.nodes.node_registration_orchestrator.services.service_introspection_topic_store import (
    ServiceIntrospectionTopicStore,
)

logger = logging.getLogger(__name__)

_WARNING_COLD_REGISTRY = "Registry cold — no nodes registered yet"


class HandlerCatalogRequest:
    """Handler for ModelTopicCatalogRequest - introspection-based catalog responder.

    Processes catalog request commands on ``request-introspection.v1`` and
    returns a snapshot of all ``onex.evt.*`` topics from the accumulated
    node introspection state.

    Dependency Injection:
        Receives ``ServiceIntrospectionTopicStore`` via constructor injection.
        The store is a shared singleton wired by the registry and also injected
        into ``HandlerNodeIntrospected`` which populates it on each introspection.

    Always-Respond Contract:
        This handler ALWAYS returns a ``ModelHandlerOutput`` with exactly one
        event (``ModelTopicCatalogResponse``). It never raises exceptions to
        the caller — error conditions are encoded in the ``warnings`` field.

    Attributes:
        _topic_store: Shared in-memory store of per-node publish topics.
    """

    def __init__(
        self,
        topic_store: ServiceIntrospectionTopicStore,
    ) -> None:
        """Initialize with the shared introspection topic store.

        Args:
            topic_store: Shared ``ServiceIntrospectionTopicStore`` instance.
                Populated by ``HandlerNodeIntrospected`` and read here.
        """
        self._topic_store = topic_store

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-catalog-request"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelTopicCatalogRequest"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns COMPUTE because the topic union and filter are pure in-memory
        operations — no external I/O is performed.
        """
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelTopicCatalogRequest],
    ) -> ModelHandlerOutput[object]:
        """Process a catalog request and return a catalog response.

        Reads the accumulated introspection state, unions all ``onex.evt.*``
        publish topics from registered nodes, and returns a
        ``ModelTopicCatalogResponse`` with the result.

        Cold-start behavior: if no nodes have registered, responds immediately
        with an empty topic list and a cold-registry warning.

        Args:
            envelope: Event envelope containing ``ModelTopicCatalogRequest``
                payload.

        Returns:
            ``ModelHandlerOutput`` with exactly one event:
            ``ModelTopicCatalogResponse``.
        """
        start_time = time.perf_counter()
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Extract correlation_id from payload if available (preferred over envelope)
        payload = envelope.payload
        if isinstance(payload, ModelTopicCatalogRequest):
            correlation_id = payload.correlation_id
            logger.debug(
                "HandlerCatalogRequest received request from requester=%s",
                payload.requester,
                extra={
                    "correlation_id": str(correlation_id),
                    "requester": payload.requester,
                },
            )
        else:
            logger.warning(
                "HandlerCatalogRequest received unexpected payload type: %s — "
                "using envelope correlation_id",
                type(payload).__name__,
                extra={"correlation_id": str(correlation_id)},
            )

        # Atomically snapshot the accumulated introspection state
        evt_topics, node_count, nodes_missing = await self._topic_store.snapshot()

        warnings: list[str] = []
        if node_count == 0:
            warnings.append(_WARNING_COLD_REGISTRY)
            logger.info(
                "HandlerCatalogRequest: registry cold (no nodes registered)",
                extra={"correlation_id": str(correlation_id)},
            )
        else:
            logger.info(
                "HandlerCatalogRequest returning %d onex.evt.* topics from %d nodes "
                "(%d missing event_bus config)",
                len(evt_topics),
                node_count,
                nodes_missing,
                extra={
                    "correlation_id": str(correlation_id),
                    "topic_count": len(evt_topics),
                    "node_count": node_count,
                    "nodes_missing_event_bus": nodes_missing,
                },
            )

        # Build catalog entries matching the omnidash TopicEntrySchema wire format.
        # ModelTopicCatalogEntry serializes to {"topic_name": ..., "topic_suffix": ..., ...}.
        # Omnidash TopicCatalogResponseSchema uses z.object({topic_name: z.string()})
        # which accepts — and ignores — extra fields. Only topic_name is required.
        topic_entries = tuple(
            ModelTopicCatalogEntry(
                topic_suffix=t,
                topic_name=t,
                partitions=1,
                publisher_count=1,  # known to exist (node published it)
            )
            for t in evt_topics
        )

        response = ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=topic_entries,
            catalog_version=1,
            node_count=node_count,
            generated_at=now,
            warnings=tuple(warnings),
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=(response,),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )


__all__: list[str] = ["HandlerCatalogRequest"]

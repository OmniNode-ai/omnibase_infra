# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent effect adapter for Consul service registration.

This module provides the IntentEffectConsulRegister adapter, which bridges
ModelPayloadConsulRegister intent payloads to actual Consul registration
operations via the HandlerConsul mixin's _register_service method.

Architecture:
    HandlerNodeIntrospected
        -> ModelPayloadConsulRegister (intent payload)
        -> IntentExecutor
        -> IntentEffectConsulRegister.execute()
        -> HandlerConsul._register_service() (Consul API)
        -> [if delta] ServiceTopicCatalog.increment_version() + emit ModelTopicCatalogChanged

    The adapter extracts service_id, service_name, tags, and health_check
    from the intent payload and delegates to the HandlerConsul for actual
    Consul agent registration.

    When the registration produces a non-empty topic delta (topics added or
    removed from the reverse index), the adapter atomically increments the
    catalog version via CAS and emits a ModelTopicCatalogChanged event to
    ``onex.evt.platform.topic-catalog-changed.v1`` (D5 design decision).

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - OMN-2314: Topic Catalog change notification + CAS versioning
    - ModelPayloadConsulRegister: Intent payload model
    - HandlerConsul: Consul handler with _register_service()
    - MixinConsulService: Mixin providing _register_service implementation
    - ServiceTopicCatalog: CAS version increment
    - ModelTopicCatalogChanged: Change notification model

.. versionadded:: 0.7.0
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.models.consul.model_consul_register_payload import (
    ModelConsulRegisterPayload,
)
from omnibase_infra.models.catalog.model_topic_catalog_changed import (
    ModelTopicCatalogChanged,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_TOPIC_CATALOG_CHANGED
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerConsul
    from omnibase_infra.handlers.models.model_consul_handler_response import (
        ModelConsulHandlerResponse,
    )
    from omnibase_infra.protocols import ProtocolEventBusLike
    from omnibase_infra.services.service_topic_catalog import ServiceTopicCatalog

logger = logging.getLogger(__name__)


class IntentEffectConsulRegister:
    """Intent effect adapter for Consul service registration.

    Bridges ModelPayloadConsulRegister intent payloads to HandlerConsul
    operations. The adapter extracts service registration fields from the
    payload and delegates to the consul handler.

    When the Consul registration results in a non-empty topic index delta
    (topics were added or removed), the adapter:

    1. Calls ``ServiceTopicCatalog.increment_version()`` (CAS, 3 retries).
       Returns ``-1`` on exhausted retries per design decision D3.
    2. Emits ``ModelTopicCatalogChanged`` with sorted delta tuples and the
       new catalog version to the configured event bus (D5).

    Thread Safety:
        This class is designed for single-threaded async use. The underlying
        HandlerConsul manages its own thread pool for synchronous Consul
        API calls.

    Attributes:
        _consul_handler: HandlerConsul for Consul service registration.
        _catalog_service: Optional ServiceTopicCatalog for CAS version increment.
        _event_bus: Optional event bus for publishing ModelTopicCatalogChanged.

    Example:
        ```python
        effect = IntentEffectConsulRegister(consul_handler=consul_handler)
        await effect.execute(payload, correlation_id=correlation_id)
        ```

        With catalog change notification:
        ```python
        effect = IntentEffectConsulRegister(
            consul_handler=consul_handler,
            catalog_service=catalog_service,
            event_bus=event_bus,
        )
        await effect.execute(payload, correlation_id=correlation_id)
        ```

    .. versionadded:: 0.7.0
    """

    def __init__(
        self,
        consul_handler: HandlerConsul,
        *,
        catalog_service: ServiceTopicCatalog | None = None,
        event_bus: ProtocolEventBusLike | None = None,
    ) -> None:
        """Initialize the Consul register intent effect.

        Args:
            consul_handler: HandlerConsul for Consul service registration.
                Must be fully initialized with a valid Consul client.
            catalog_service: Optional ServiceTopicCatalog for CAS version
                increment. When None, both the CAS version increment and the
                change event emission are skipped entirely, even if the topic
                delta is non-empty.
            event_bus: Optional event bus for publishing ModelTopicCatalogChanged.
                When None, both the CAS version increment and the change event
                emission are skipped entirely, even if the topic delta is
                non-empty. Both catalog_service and event_bus must be non-None
                for any catalog change notification to occur.
        """
        self._consul_handler = consul_handler
        self._catalog_service = catalog_service
        self._event_bus = event_bus

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute a Consul service registration from an intent payload.

        Extracts service registration fields from the payload and delegates
        to HandlerConsul._register_service() via the mixin's handle() method.

        After a successful registration, if the topic index delta is non-empty
        and both ``catalog_service`` and ``event_bus`` are configured, this
        method increments the catalog version and emits
        ``ModelTopicCatalogChanged``.

        The payload fields are converted to the dict format expected by
        the HandlerConsul mixin's _register_service method.

        Args:
            payload: The ModelPayloadConsulRegister intent payload.
                Validated via isinstance at entry.
            correlation_id: Optional correlation ID for tracing.
                Falls back to payload.correlation_id if not provided.

        Raises:
            RuntimeHostError: If the Consul registration fails.
        """
        # Compute effective correlation_id before type checks so error contexts
        # always carry a non-None ID, preserving any ID from the payload when
        # available and falling back to uuid4() only as a last resort.
        effective_correlation_id = (
            correlation_id or getattr(payload, "correlation_id", None) or uuid4()
        )

        if not isinstance(payload, ModelPayloadConsulRegister):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="intent_effect_consul_register",
            )
            raise RuntimeHostError(
                f"Expected ModelPayloadConsulRegister, got {type(payload).__name__}",
                context=context,
            )

        try:
            # Build the payload dict expected by HandlerConsul._register_service
            register_payload: dict[str, object] = {
                "name": payload.service_name,
                "service_id": payload.service_id,
                "tags": payload.tags,
            }

            if payload.node_id is not None:
                register_payload["node_id"] = payload.node_id

            if payload.address is not None:
                register_payload["address"] = payload.address
            if payload.port is not None:
                register_payload["port"] = payload.port

            if payload.health_check is not None:
                register_payload["check"] = payload.health_check

            # Store event_bus_config if present (for topic routing lookups)
            if payload.event_bus_config is not None:
                register_payload["event_bus_config"] = (
                    payload.event_bus_config.model_dump()
                )

            # Delegate to HandlerConsul via its execute() method.
            # The handler routes internally based on the "operation" key and
            # returns a ModelHandlerOutput whose result payload carries the
            # topic index delta (topics_added, topics_removed).
            envelope: dict[str, object] = {
                "operation": "consul.register",
                "payload": register_payload,
                "correlation_id": effective_correlation_id,
                "envelope_id": str(uuid4()),
            }
            handler_output = await self._consul_handler.execute(envelope)

            logger.info(
                "Consul registration executed: service_id=%s service_name=%s "
                "correlation_id=%s",
                payload.service_id,
                payload.service_name,
                str(effective_correlation_id),
            )

            # --- 3.3 Emit change notification if topic delta is non-empty ---
            await self._maybe_emit_catalog_changed(
                handler_output=handler_output,
                node_id=payload.node_id,
                correlation_id=effective_correlation_id,
            )

        except RuntimeHostError:
            raise
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="intent_effect_consul_register",
            )
            logger.warning(
                "Consul registration intent failed: error=%s correlation_id=%s",
                sanitize_error_message(e),
                str(effective_correlation_id),
                extra={
                    "error_type": type(e).__name__,
                    "service_id": payload.service_id,
                },
            )
            raise RuntimeHostError(
                "Failed to execute Consul registration intent",
                context=context,
            ) from e

    async def _maybe_emit_catalog_changed(
        self,
        handler_output: ModelHandlerOutput[ModelConsulHandlerResponse],
        node_id: str | None,
        correlation_id: UUID,
    ) -> None:
        """Emit ModelTopicCatalogChanged when the topic delta is non-empty.

        Reads the topic delta from the handler output's result payload.
        If both ``topics_added`` and ``topics_removed`` are empty the method
        returns immediately without any side effects.

        When a non-empty delta is detected:
        1. Calls ``ServiceTopicCatalog.increment_version()`` (CAS with retries).
        2. Builds ``ModelTopicCatalogChanged`` with sorted delta tuples and the
           new catalog version (``-1`` signals a CAS failure per D3).
        3. Publishes the change event to the event bus.

        Errors during catalog version increment or event publishing are caught
        and logged at WARNING level so that a notification failure does not
        abort the Consul registration.

        Args:
            handler_output: Return value from ``HandlerConsul.execute()``.
                Typed as ``ModelHandlerOutput[ModelConsulHandlerResponse]``.
            node_id: Node identifier for the ``trigger_node_id`` field.
            correlation_id: Correlation ID for tracing.
        """
        if self._catalog_service is None or self._event_bus is None:
            return

        if not hasattr(handler_output, "result"):
            return

        # Extract delta from handler output result using typed access.
        # result is ModelConsulHandlerResponse | None; payload.data is a
        # discriminated ConsulPayload union — only ModelConsulRegisterPayload
        # carries topics_added / topics_removed.
        topics_added: frozenset[str] = frozenset()
        topics_removed: frozenset[str] = frozenset()

        result = handler_output.result
        if result is not None:
            data = result.payload.data
            if isinstance(data, ModelConsulRegisterPayload):
                topics_added = data.topics_added
                topics_removed = data.topics_removed

        if not topics_added and not topics_removed:
            # No change - nothing to emit
            return

        logger.debug(
            "Topic delta detected: +%d topics, -%d topics (correlation_id=%s)",
            len(topics_added),
            len(topics_removed),
            str(correlation_id),
            extra={
                "topics_added": sorted(topics_added),
                "topics_removed": sorted(topics_removed),
            },
        )

        try:
            # CAS version increment (3 retries, -1 on failure per D3)
            new_version = await self._catalog_service.increment_version(correlation_id)

            if new_version == -1:
                # CAS retries exhausted; catalog_version will be clamped to 0
                # (max(-1, 0)) in the emitted event. Consumers can distinguish
                # this from a genuine version-0 catalog via the cas_failure=True
                # field on the emitted ModelTopicCatalogChanged event.
                logger.warning(
                    "CAS version increment exhausted retries; emitting "
                    "ModelTopicCatalogChanged with catalog_version=0 and "
                    "cas_failure=True — inspect the cas_failure field on the "
                    "emitted event to distinguish this from a genuine "
                    "version-0 catalog (correlation_id=%s)",
                    str(correlation_id),
                    extra={
                        "topics_added": sorted(topics_added),
                        "topics_removed": sorted(topics_removed),
                        "trigger_node_id": node_id,
                    },
                )

            # Determine trigger_reason based on delta content
            if topics_added and topics_removed:
                trigger_reason = "capability_change"
            elif topics_added:
                trigger_reason = "registration"
            else:
                trigger_reason = "deregistration"

            changed_event = ModelTopicCatalogChanged(
                correlation_id=correlation_id,
                catalog_version=max(new_version, 0),
                topics_added=tuple(sorted(topics_added)),
                topics_removed=tuple(sorted(topics_removed)),
                trigger_node_id=node_id,
                trigger_reason=trigger_reason,
                changed_at=datetime.now(UTC),
                cas_failure=(new_version == -1),
            )

            change_envelope: ModelEventEnvelope[ModelTopicCatalogChanged] = (
                ModelEventEnvelope(
                    payload=changed_event,
                    envelope_timestamp=datetime.now(UTC),
                    correlation_id=correlation_id,
                    envelope_id=uuid4(),
                )
            )

            # publish_envelope(envelope, topic) — envelope is first, topic is second
            await self._event_bus.publish_envelope(
                change_envelope,
                SUFFIX_TOPIC_CATALOG_CHANGED,
            )

            logger.info(
                "Emitted ModelTopicCatalogChanged: version=%d +%d -%d "
                "(correlation_id=%s)",
                changed_event.catalog_version,
                len(topics_added),
                len(topics_removed),
                str(correlation_id),
                extra={
                    "catalog_version": changed_event.catalog_version,
                    "topics_added": sorted(topics_added),
                    "topics_removed": sorted(topics_removed),
                    "trigger_reason": trigger_reason,
                    "trigger_node_id": node_id,
                },
            )

        except RuntimeHostError as emit_err:
            # Catalog change notification is best-effort: do not fail the
            # registration if the notification cannot be emitted due to an
            # infrastructure-level failure (e.g. Kafka unavailable, connection
            # error, timeout).  Programming errors such as
            # ``pydantic.ValidationError`` are intentionally NOT caught here
            # so that bugs in event construction surface immediately.
            logger.warning(
                "Failed to emit catalog changed event: %s (correlation_id=%s)",
                sanitize_error_message(emit_err),
                str(correlation_id),
                extra={"error_type": type(emit_err).__name__},
            )


__all__: list[str] = ["IntentEffectConsulRegister"]

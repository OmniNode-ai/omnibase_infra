# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Startup introspection of the runtime host (OMN-1930, OMN-21590).

After handlers are wired the runtime announces its presence. Two protections
apply: jitter (a random delay so many nodes restarting together do not publish
at once) and throttling (skip when the last introspection was too recent).

The runtime's own introspection service always publishes the runtime's event
first. With announcements, one more introspection event per ``announce`` record
is built from the announcement and the runtime's identity and published on the
introspection topic, so the registry reads the announced nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from uuid import UUID

from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumIntrospectionReason,
)
from omnibase_infra.models.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra.models.model_runtime_introspection_startup_request import (
    ModelRuntimeIntrospectionStartupRequest,
)
from omnibase_infra.models.model_runtime_introspection_startup_result import (
    ModelRuntimeIntrospectionStartupResult,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.protocols.protocol_node_introspection import (
    ProtocolNodeIntrospection,
)
from omnibase_infra.topics import SUFFIX_NODE_INTROSPECTION

logger = logging.getLogger(__name__)


class HandlerRuntimeIntrospectionStartup:
    """Publish the runtime's startup introspection with jitter and throttling."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        request: ModelRuntimeIntrospectionStartupRequest,
        *,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
        event_bus: ProtocolEventBusLike | None,
        build_events: Callable[
            [tuple[ModelNodeAnnouncement, ...], ModelRuntimeIdentity],
            tuple[ModelNodeIntrospectionEvent, ...],
        ],
    ) -> ModelRuntimeIntrospectionStartupResult:
        """Publish the startup introspection unless disabled or throttled."""
        correlation_id = request.correlation_id
        last_time = request.last_introspection_time
        not_published = ModelRuntimeIntrospectionStartupResult(
            published=False, last_introspection_time=last_time
        )

        # Guard: Skip if introspection not configured
        if service is None:
            logger.debug("Introspection service not configured, skipping")
            return not_published

        if not config.enabled:
            logger.debug("Introspection disabled by config, skipping")
            return not_published

        # Throttle check: Skip if last introspection was too recent
        if last_time is not None:
            elapsed = time.monotonic() - last_time
            if elapsed < config.throttle_min_interval_s:
                logger.debug(
                    "Introspection throttled",
                    extra={
                        "elapsed_seconds": elapsed,
                        "throttle_min_interval_s": (config.throttle_min_interval_s),
                    },
                )
                return not_published

        # Apply jitter: Random delay to prevent thundering herd
        jitter_ms = random.randint(0, config.jitter_max_ms)
        if jitter_ms > 0:
            logger.debug(
                "Applying introspection jitter",
                extra={"jitter_ms": jitter_ms},
            )
            await asyncio.sleep(jitter_ms / 1000.0)

        # Publish introspection event
        try:
            await service.publish_introspection(
                reason=EnumIntrospectionReason.STARTUP,
                correlation_id=correlation_id,
            )
            if request.announcements and event_bus is not None:
                await self._publish_announced(service, request, event_bus, build_events)

            # Record time for throttle tracking immediately after publish
            last_time = time.monotonic()

            logger.info(
                "Startup introspection published",
                extra={
                    "correlation_id": str(correlation_id),
                    "jitter_ms": jitter_ms,
                },
            )

        except Exception as e:
            # Log the failure but do not fail startup - introspection is optional
            logger.exception(
                "Failed to publish startup introspection",
                extra={
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )
            return not_published
        return ModelRuntimeIntrospectionStartupResult(
            published=True, last_introspection_time=last_time
        )

    async def start_heartbeat(
        self,
        correlation_id: UUID,
        *,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
    ) -> None:
        """Start the heartbeat task for periodic liveness announcements."""
        if service is None or not config.enabled:
            return
        # Separate try block: heartbeat failure shouldn't affect throttle tracking
        try:
            await service.start_heartbeat_task()
        except Exception as e:
            logger.exception(
                "Failed to start heartbeat task",
                extra={
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )

    async def _publish_announced(
        self,
        service: ProtocolNodeIntrospection,
        request: ModelRuntimeIntrospectionStartupRequest,
        event_bus: ProtocolEventBusLike,
        build_events: Callable[
            [tuple[ModelNodeAnnouncement, ...], ModelRuntimeIdentity],
            tuple[ModelNodeIntrospectionEvent, ...],
        ],
    ) -> None:
        """Publish one introspection event per announce record."""
        get_data = getattr(service, "get_introspection_data", None)
        if not callable(get_data):
            raise TypeError("introspection service cannot report its identity")
        data = await get_data()
        if not isinstance(data, ModelNodeIntrospectionEvent):
            raise TypeError("introspection service cannot report its identity")
        identity = ModelRuntimeIdentity.from_introspection_event(data)
        for event in build_events(request.announcements, identity):
            await self._publish_event(event, identity.node_id, event_bus)

    async def _publish_event(
        self,
        event: ModelNodeIntrospectionEvent,
        node_id: UUID,
        event_bus: ProtocolEventBusLike,
    ) -> None:
        """Publish one introspection event the way the introspection mixin does."""
        if hasattr(event_bus, "publish_envelope"):
            envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                payload=event,
                correlation_id=event.correlation_id,
                tenant_id=None,
            )
            await event_bus.publish_envelope(
                envelope=envelope, topic=SUFFIX_NODE_INTROSPECTION
            )
        else:
            await event_bus.publish(
                topic=SUFFIX_NODE_INTROSPECTION,
                key=str(node_id).encode("utf-8"),
                value=json.dumps(event.model_dump(mode="json")).encode("utf-8"),
            )


__all__ = ["HandlerRuntimeIntrospectionStartup"]

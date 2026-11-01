# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Startup introspection of the runtime host (OMN-1930, OMN-21590).

After handlers are wired the runtime announces its presence. Two protections
apply: jitter (a random delay so many nodes restarting together do not publish
at once) and throttling (skip when the last introspection was too recent).

With no node announcements the runtime's own introspection service publishes its
event. With announcements, one introspection event per ``announce`` record is
built from the announcement and the runtime's identity and published on the
introspection topic, so the registry reads the announced nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Final
from uuid import UUID

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumIntrospectionReason,
)
from omnibase_infra.handlers.handler_registration_from_announcement import (
    HandlerRegistrationFromAnnouncement,
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
from omnibase_infra.protocols.protocol_introspection_data_source import (
    ProtocolIntrospectionDataSource,
)
from omnibase_infra.protocols.protocol_node_introspection import (
    ProtocolNodeIntrospection,
)
from omnibase_infra.topics import SUFFIX_NODE_INTROSPECTION

logger = logging.getLogger(__name__)

# What publishing an introspection event or starting the heartbeat can raise.
# Introspection is optional: these are logged and startup goes on.
_INTROSPECTION_FAILURES: Final = (
    ModelOnexError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    LookupError,
    TimeoutError,
)


class HandlerRuntimeIntrospectionStartup:
    """Publish the runtime's startup introspection with jitter and throttling."""

    def __init__(
        self,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
        event_bus: ProtocolEventBusLike | None = None,
        topic: str = SUFFIX_NODE_INTROSPECTION,
    ) -> None:
        self._service = service
        self._config = config
        self._event_bus = event_bus
        self._topic = topic

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelRuntimeIntrospectionStartupRequest
    ) -> ModelRuntimeIntrospectionStartupResult:
        """Publish the startup introspection unless disabled or throttled."""
        correlation_id = request.correlation_id
        last_time = request.last_introspection_time
        not_published = ModelRuntimeIntrospectionStartupResult(
            published=False, last_introspection_time=last_time
        )

        # Guard: Skip if introspection not configured
        if self._service is None:
            logger.debug("Introspection service not configured, skipping")
            return not_published

        if not self._config.enabled:
            logger.debug("Introspection disabled by config, skipping")
            return not_published

        # Throttle check: Skip if last introspection was too recent
        if last_time is not None:
            elapsed = time.monotonic() - last_time
            if elapsed < self._config.throttle_min_interval_s:
                logger.debug(
                    "Introspection throttled",
                    extra={
                        "elapsed_seconds": elapsed,
                        "throttle_min_interval_s": (
                            self._config.throttle_min_interval_s
                        ),
                    },
                )
                return not_published

        # Apply jitter: Random delay to prevent thundering herd
        jitter_ms = random.randint(0, self._config.jitter_max_ms)
        if jitter_ms > 0:
            logger.debug(
                "Applying introspection jitter",
                extra={"jitter_ms": jitter_ms},
            )
            await asyncio.sleep(jitter_ms / 1000.0)

        # Publish introspection event
        try:
            if request.announcements and self._event_bus is not None:
                await self._publish_announced(self._service, request)
            else:
                await self._service.publish_introspection(
                    reason=EnumIntrospectionReason.STARTUP,
                    correlation_id=correlation_id,
                )

            # Record time for throttle tracking immediately after publish
            last_time = time.monotonic()

            logger.info(
                "Startup introspection published",
                extra={
                    "correlation_id": str(correlation_id),
                    "jitter_ms": jitter_ms,
                },
            )

        except _INTROSPECTION_FAILURES as e:
            # Log warning but don't fail startup - introspection is optional
            logger.warning(
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

    async def start_heartbeat(self, correlation_id: UUID) -> None:
        """Start the heartbeat task for periodic liveness announcements."""
        if self._service is None or not self._config.enabled:
            return
        # Separate try block: heartbeat failure shouldn't affect throttle tracking
        try:
            await self._service.start_heartbeat_task()
        except _INTROSPECTION_FAILURES as e:
            logger.warning(
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
    ) -> None:
        """Publish one introspection event per announce record."""
        if not isinstance(service, ProtocolIntrospectionDataSource):
            raise TypeError("introspection service cannot report its identity")
        identity = ModelRuntimeIdentity.from_introspection_event(
            await service.get_introspection_data()
        )
        for event in HandlerRegistrationFromAnnouncement.build_events(
            request.announcements, identity
        ):
            await self._publish_event(event, identity.node_id)

    async def _publish_event(
        self,
        event: ModelNodeIntrospectionEvent,
        node_id: UUID,
    ) -> None:
        """Publish one introspection event the way the introspection mixin does."""
        event_bus = self._event_bus
        if event_bus is None:
            raise ValueError("no event bus configured")
        if hasattr(event_bus, "publish_envelope"):
            envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                payload=event,
                correlation_id=event.correlation_id,
                tenant_id=None,
            )
            await event_bus.publish_envelope(envelope=envelope, topic=self._topic)
        else:
            await event_bus.publish(
                topic=self._topic,
                key=str(node_id).encode("utf-8"),
                value=json.dumps(event.model_dump(mode="json")).encode("utf-8"),
            )


__all__ = ["HandlerRuntimeIntrospectionStartup"]

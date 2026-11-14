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
from typing import TypeVar
from uuid import UUID

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumIntrospectionReason,
    EnumRuntimeHeartbeatRefusalReason,
)
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.model_runtime_heartbeat_refusal import (
    ModelRuntimeHeartbeatRefusal,
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

_ServiceT = TypeVar("_ServiceT")


class RuntimeHeartbeatRequiredError(RuntimeHostError):
    """A runtime refused to start because its heartbeat could not run (OMN-24496).

    Raised at the refusal points of this handler and carries the refusal as a
    ``ModelRuntimeHeartbeatRefusal`` on ``.refusal``. The refusal records how
    many times this process has refused for this reason (``count``), so a
    crash loop that keeps hitting the same wall is visible in the refusal
    stream rather than only in log volume.
    """

    def __init__(
        self,
        message: str,
        refusal: ModelRuntimeHeartbeatRefusal,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        self.refusal = refusal
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_STATE,
            context=context,
            **extra_context,
        )


class HandlerRuntimeIntrospectionStartup:
    """Publish the runtime's startup introspection with jitter and throttling.

    The heartbeat is mandatory (OMN-24496): a runtime that cannot build its
    introspection service or start its heartbeat task refuses to start with a
    typed :class:`RuntimeHeartbeatRequiredError` instead of being announced
    without a heartbeat.
    """

    # Per-reason refusal counters for this process. A runtime that keeps
    # hitting the same wall in a crash loop increments its reason's counter on
    # every refused start, so the refusal stream shows the loop. Tests reset
    # this via :meth:`reset_refusal_counts`.
    _refusal_counts: dict[EnumRuntimeHeartbeatRefusalReason, int] = {}

    @classmethod
    def refusal_count(cls, reason: EnumRuntimeHeartbeatRefusalReason) -> int:
        """Return how many refusals of ``reason`` this process has made."""
        return cls._refusal_counts.get(reason, 0)

    @classmethod
    def reset_refusal_counts(cls) -> None:
        """Clear the per-reason refusal counters (test isolation)."""
        cls._refusal_counts.clear()

    def refusal_error(
        self,
        *,
        reason: EnumRuntimeHeartbeatRefusalReason,
        runtime_name: str,
        detail: str,
        correlation_id: UUID,
    ) -> RuntimeHeartbeatRequiredError:
        """Build the typed refusal for a runtime that cannot heartbeat.

        Increments the per-reason refusal counter, logs the refusal at error
        level, and returns the error the caller raises (``from exc``).

        Args:
            reason: Why the runtime cannot heartbeat
            runtime_name: Name of the runtime being refused
            detail: Human-readable failure detail behind the refusal
            correlation_id: Correlation ID of the refused startup attempt

        Returns:
            The typed refusal error carrying the counted refusal
        """
        count = self._refusal_counts.get(reason, 0) + 1
        self._refusal_counts[reason] = count
        refusal = ModelRuntimeHeartbeatRefusal(
            runtime_name=runtime_name,
            reason=reason,
            detail=detail,
            correlation_id=correlation_id,
            count=count,
        )
        logger.error(
            "Runtime heartbeat is mandatory: refusing to start",
            extra={"refusal": refusal.model_dump(mode="json")},
        )
        return RuntimeHeartbeatRequiredError(
            f"Runtime heartbeat is mandatory: refusing to start runtime "
            f"'{runtime_name}' ({reason.value}): {detail}",
            refusal=refusal,
        )

    def build_required_service(
        self,
        *,
        runtime_name: str,
        correlation_id: UUID,
        build: Callable[[], _ServiceT],
    ) -> _ServiceT:
        """Build the introspection service, refusing to start without one.

        Args:
            runtime_name: Name of the runtime the service is built for
            correlation_id: Correlation ID of the startup attempt
            build: Zero-argument callable that builds the service

        Returns:
            The built introspection service

        Raises:
            RuntimeHeartbeatRequiredError: The runtime name is empty, or the
                service could not be built
        """
        if not runtime_name.strip():
            raise self.refusal_error(
                reason=EnumRuntimeHeartbeatRefusalReason.RUNTIME_NAME_EMPTY,
                runtime_name=runtime_name,
                detail="runtime_name is empty or whitespace",
                correlation_id=correlation_id,
            )
        try:
            return build()
        except Exception as exc:
            raise self.refusal_error(
                reason=(
                    EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_UNBUILDABLE
                ),
                runtime_name=runtime_name,
                detail=f"{type(exc).__name__}: {exc}",
                correlation_id=correlation_id,
            ) from exc

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

        # OMN-24496: The heartbeat is mandatory. It starts right after the
        # runtime's own presence is published and BEFORE any announced node is
        # published, outside the try above so a refusal propagates: no node is
        # announced for a runtime whose heartbeat cannot start.
        await self.start_heartbeat(correlation_id, service=service, config=config)

        # Runtime presence has succeeded. Optional announced-node publishing
        # must not erase its throttle timestamp or suppress its heartbeat.
        if request.announcements and event_bus is not None:
            try:
                await self._publish_announced(service, request, event_bus, build_events)
            except Exception as e:
                logger.exception(
                    "Failed to publish announced node introspection",
                    extra={
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )
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
        """Start the heartbeat task for periodic liveness announcements.

        The heartbeat is mandatory for kernel runtimes (OMN-24496): a failure
        to start it refuses the runtime's startup. Service ``None`` or
        introspection disabled by config returns silently — those are not
        kernel runtimes.
        """
        if service is None or not config.enabled:
            return
        try:
            await service.start_heartbeat_task()
        except Exception as exc:
            name = getattr(service, "_introspection_node_name", None)
            raise self.refusal_error(
                reason=EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED,
                runtime_name=name if isinstance(name, str) and name else "unknown",
                detail=f"{type(exc).__name__}: {exc}",
                correlation_id=correlation_id,
            ) from exc

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


__all__ = ["HandlerRuntimeIntrospectionStartup", "RuntimeHeartbeatRequiredError"]

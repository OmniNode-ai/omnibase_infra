# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build loop domain wiring for MessageDispatchEngine integration.

Registers the build loop handler in the DI container and wires the
dispatcher into the MessageDispatchEngine for event-driven routing.

Related:
    - OMN-7319: node_autonomous_loop_orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypedDict
from uuid import UUID

from omnibase_core.enums import EnumInjectionScope, EnumMessageCategory

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.protocols.event_bus import ProtocolEventBusPublisher
    from omnibase_infra.runtime import MessageDispatchEngine

logger = logging.getLogger(__name__)

ROUTE_ID_BUILD_LOOP_START = "route.build-loop.build-loop-start"


def _make_publisher(
    event_bus: ProtocolEventBusPublisher,
) -> Callable[..., Awaitable[bool]]:
    """Create a publisher callback that bridges ProtocolEventBusPublisher to
    the HandlerLoopOrchestrator's expected signature.

    The orchestrator calls ``publisher(event_type=..., payload=...,
    correlation_id=..., topic=...)`` and expects a ``bool`` return.
    """

    async def _publish(
        *,
        event_type: str,
        payload: dict[str, object],
        correlation_id: UUID,
        topic: str,
    ) -> bool:
        try:
            value = json.dumps(
                {
                    "event_type": event_type,
                    "correlation_id": str(correlation_id),
                    "payload": payload,
                },
            ).encode()
            key = str(correlation_id).encode()
            await event_bus.publish(topic, key, value)
            return True
        except Exception:  # noqa: BLE001 — caller logs the False return
            logger.warning(
                "Event bus publish failed (topic=%s, correlation_id=%s)",
                topic,
                correlation_id,
                exc_info=True,
            )
            return False

    return _publish


class WiringResult(TypedDict):
    services: list[str]
    status: str


async def wire_build_loop_handlers(
    container: ModelONEXContainer,
) -> WiringResult:
    """Register build loop handlers with the container.

    Registers:
    - HandlerLoopOrchestrator (6-phase build loop orchestrator)

    Args:
        container: ONEX container instance to register services in.

    Returns:
        WiringResult with list of registered service names.
    """
    from omnibase_core.protocols.event_bus import ProtocolEventBusPublisher
    from omnibase_infra.nodes.node_autonomous_loop_orchestrator.handlers.handler_loop_orchestrator import (
        HandlerLoopOrchestrator,
    )

    services_registered: list[str] = []

    linear_api_key = os.environ.get("LINEAR_API_KEY")

    # Resolve event bus publisher so the orchestrator can publish delegation
    # payloads to Kafka during the BUILD phase.
    publisher_cb = None
    if container.service_registry is not None:
        try:
            # NOTE: service_kernel registers the concrete EventBus under the
            # ProtocolEventBusPublisher key; mypy cannot verify Protocol-based DI.
            event_bus: ProtocolEventBusPublisher = (
                await container.service_registry.resolve_service(
                    ProtocolEventBusPublisher  # type: ignore[type-abstract]
                )
            )
            publisher_cb = _make_publisher(event_bus)
            logger.debug("Resolved ProtocolEventBusPublisher for build loop")
        except Exception:  # noqa: BLE001 — degrade to filesystem fallback
            logger.warning(
                "Could not resolve ProtocolEventBusPublisher — "
                "build loop will use filesystem fallback",
                exc_info=True,
            )

    handler = HandlerLoopOrchestrator(
        publisher=publisher_cb,
        linear_api_key=linear_api_key,
    )

    if container.service_registry is not None:
        await container.service_registry.register_instance(
            interface=HandlerLoopOrchestrator,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
            metadata={
                "description": "Build loop orchestrator (6-phase FSM)",
            },
        )
    services_registered.append("HandlerLoopOrchestrator")
    logger.debug("Registered HandlerLoopOrchestrator in container")

    return WiringResult(services=services_registered, status="success")


async def wire_build_loop_dispatchers(
    container: ModelONEXContainer,
    engine: MessageDispatchEngine,
    correlation_id: UUID | None = None,
) -> dict[str, list[str] | str]:
    """Wire build loop dispatchers into MessageDispatchEngine.

    Creates dispatcher adapters for the build loop handler and registers
    them with the MessageDispatchEngine.

    Args:
        container: ONEX container with registered handlers.
        engine: MessageDispatchEngine to register dispatchers with.
        correlation_id: Optional correlation ID for error tracking.

    Returns:
        Summary dict with dispatchers, routes, and status.
    """
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_infra.nodes.node_autonomous_loop_orchestrator.dispatchers.dispatcher_build_loop_start import (
        DispatcherBuildLoopStart,
    )
    from omnibase_infra.nodes.node_autonomous_loop_orchestrator.handlers.handler_loop_orchestrator import (
        HandlerLoopOrchestrator,
    )

    dispatchers_registered: list[str] = []
    routes_registered: list[str] = []

    handler: HandlerLoopOrchestrator = await container.service_registry.resolve_service(
        HandlerLoopOrchestrator
    )

    dispatcher = DispatcherBuildLoopStart(handler)
    engine.register_dispatcher(
        dispatcher_id=dispatcher.dispatcher_id,
        dispatcher=dispatcher.handle,
        category=dispatcher.category,
        message_types=dispatcher.message_types,
    )
    dispatchers_registered.append(dispatcher.dispatcher_id)

    route = ModelDispatchRoute(
        route_id=ROUTE_ID_BUILD_LOOP_START,
        topic_pattern="*.cmd.*.build-loop-start.*",
        message_category=EnumMessageCategory.COMMAND,
        dispatcher_id=dispatcher.dispatcher_id,
        message_type="omnibase-infra.build-loop-start",
    )
    engine.register_route(route)
    routes_registered.append(route.route_id)

    logger.info(
        "Build loop dispatchers wired: %s (correlation_id=%s)",
        dispatchers_registered,
        correlation_id,
    )

    return {
        "dispatchers": dispatchers_registered,
        "routes": routes_registered,
        "status": "success",
    }

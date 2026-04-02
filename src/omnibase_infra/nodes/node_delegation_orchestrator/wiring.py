# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Delegation domain wiring for MessageDispatchEngine integration.

Registers delegation handlers in the DI container and wires dispatchers
into the MessageDispatchEngine for event-driven routing.

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict
from uuid import UUID

from omnibase_core.enums import EnumInjectionScope

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
    from omnibase_infra.runtime import MessageDispatchEngine

logger = logging.getLogger(__name__)

# Route IDs for delegation dispatchers
ROUTE_ID_DELEGATION_REQUEST = "route.delegation.delegation-request"
ROUTE_ID_ROUTING_DECISION = "route.delegation.routing-decision"
ROUTE_ID_QUALITY_GATE_RESULT = "route.delegation.quality-gate-result"


class WiringResult(TypedDict):
    services: list[str]
    status: str


async def wire_delegation_handlers(
    container: ModelONEXContainer,
) -> WiringResult:
    """Register delegation handlers with the container.

    Registers:
    - HandlerDelegationWorkflow (orchestrator FSM)
    - HandlerDelegationRouting (routing reducer — pure function, stateless)
    - HandlerQualityGate (quality gate reducer — pure function, stateless)

    Args:
        container: ONEX container instance to register services in.

    Returns:
        WiringResult with list of registered service names.
    """
    from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
        HandlerDelegationWorkflow,
    )
    from omnibase_infra.nodes.node_delegation_quality_gate_reducer.handlers.handler_quality_gate import (
        delta as quality_gate_delta,
    )
    from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
        delta as routing_delta,
    )

    services_registered: list[str] = []

    # 1. HandlerDelegationWorkflow — stateful orchestrator FSM
    workflow_handler = HandlerDelegationWorkflow()
    if container.service_registry is not None:
        await container.service_registry.register_instance(
            interface=HandlerDelegationWorkflow,
            instance=workflow_handler,
            scope=EnumInjectionScope.GLOBAL,
            metadata={
                "description": "Delegation workflow orchestrator (FSM)",
            },
        )
    services_registered.append("HandlerDelegationWorkflow")
    logger.debug("Registered HandlerDelegationWorkflow in container")

    return WiringResult(services=services_registered, status="success")


async def wire_delegation_dispatchers(
    container: ModelONEXContainer,
    engine: MessageDispatchEngine,
    correlation_id: UUID | None = None,
    event_bus: ProtocolEventBus | None = None,
) -> dict[str, list[str] | str]:
    """Wire delegation dispatchers into MessageDispatchEngine.

    Creates dispatcher adapters for the delegation handler and registers
    them with the MessageDispatchEngine.

    Args:
        container: ONEX container with registered handlers.
        engine: MessageDispatchEngine to register dispatchers with.
        correlation_id: Optional correlation ID for error tracking.
        event_bus: Optional event bus for output event publishing.

    Returns:
        Summary dict with dispatchers, routes, and status.
    """
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_infra.nodes.node_delegation_orchestrator.dispatchers.dispatcher_delegation_request import (
        DispatcherDelegationRequest,
    )
    from omnibase_infra.nodes.node_delegation_orchestrator.dispatchers.dispatcher_quality_gate_result import (
        DispatcherQualityGateResult,
    )
    from omnibase_infra.nodes.node_delegation_orchestrator.dispatchers.dispatcher_routing_decision import (
        DispatcherRoutingDecision,
    )
    from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
        HandlerDelegationWorkflow,
    )

    dispatchers_registered: list[str] = []
    routes_registered: list[str] = []

    # Resolve the workflow handler from the container
    handler: HandlerDelegationWorkflow = (
        await container.service_registry.resolve_service(HandlerDelegationWorkflow)
    )

    # 1. DispatcherDelegationRequest — handles incoming delegation commands
    dispatcher_request = DispatcherDelegationRequest(handler, event_bus=event_bus)
    engine.register_dispatcher(
        dispatcher_id=dispatcher_request.dispatcher_id,
        dispatcher=dispatcher_request.handle,
        category=dispatcher_request.category,
        message_types=dispatcher_request.message_types,
    )
    dispatchers_registered.append(dispatcher_request.dispatcher_id)

    # 2. DispatcherRoutingDecision — handles routing decisions from reducer
    dispatcher_routing = DispatcherRoutingDecision(handler, event_bus=event_bus)
    engine.register_dispatcher(
        dispatcher_id=dispatcher_routing.dispatcher_id,
        dispatcher=dispatcher_routing.handle,
        category=dispatcher_routing.category,
        message_types=dispatcher_routing.message_types,
    )
    dispatchers_registered.append(dispatcher_routing.dispatcher_id)

    # 3. DispatcherQualityGateResult — handles quality gate results
    dispatcher_gate = DispatcherQualityGateResult(handler, event_bus=event_bus)
    engine.register_dispatcher(
        dispatcher_id=dispatcher_gate.dispatcher_id,
        dispatcher=dispatcher_gate.handle,
        category=dispatcher_gate.category,
        message_types=dispatcher_gate.message_types,
    )
    dispatchers_registered.append(dispatcher_gate.dispatcher_id)

    logger.info(
        "Delegation dispatchers wired: %s (correlation_id=%s)",
        dispatchers_registered,
        correlation_id,
    )

    return {
        "dispatchers": dispatchers_registered,
        "routes": routes_registered,
        "status": "success",
    }

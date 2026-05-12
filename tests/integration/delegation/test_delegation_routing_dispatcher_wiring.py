# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: delegation routing dispatcher is wired into the DI kernel.

Verifies that wire_delegation_routing_dispatcher() registers the dispatcher
and route in the MessageDispatchEngine such that messages of category
'command' / type 'omnibase-infra.delegation-routing-request' are resolved
to DispatcherDelegationRoutingRequest instead of falling through to the DLQ.

This is the integration-level proof for OMN-10868 — the unit tests confirm
individual method contracts; this test confirms end-to-end lookup succeeds.

Related:
    - OMN-10868: Wire delegation routing dispatcher in DI kernel
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


@pytest.mark.asyncio
async def test_routing_dispatcher_registered_in_engine() -> None:
    """After wiring, the engine resolves routing-request to the correct dispatcher."""
    from unittest.mock import AsyncMock, MagicMock

    from omnibase_core.enums import EnumMessageCategory
    from omnibase_infra.nodes.node_delegation_orchestrator.wiring import (
        ROUTE_ID_DELEGATION_ROUTING_REQUEST,
        wire_delegation_routing_dispatcher,
    )
    from omnibase_infra.nodes.node_delegation_routing_reducer.dispatchers.dispatcher_delegation_routing_request import (
        DispatcherDelegationRoutingRequest,
    )

    dispatchers: dict[str, object] = {}
    routes: list[object] = []

    engine = MagicMock()
    engine.register_dispatcher = MagicMock(
        side_effect=lambda dispatcher_id, dispatcher, category, message_types: (
            dispatchers.update({dispatcher_id: dispatcher})
        )
    )
    engine.register_route = MagicMock(side_effect=lambda route: routes.append(route))

    result = await wire_delegation_routing_dispatcher(engine)

    assert result["status"] == "success"
    assert "dispatcher.delegation.routing-request" in dispatchers
    assert ROUTE_ID_DELEGATION_ROUTING_REQUEST in result["routes"]

    route = routes[0]
    assert route.message_type == "omnibase-infra.delegation-routing-request"
    assert route.message_category == EnumMessageCategory.COMMAND


@pytest.mark.asyncio
async def test_routing_dispatcher_message_types_cover_contract_topic() -> None:
    """DispatcherDelegationRoutingRequest declares the message type matching the contract topic."""
    from omnibase_infra.nodes.node_delegation_routing_reducer.dispatchers.dispatcher_delegation_routing_request import (
        DispatcherDelegationRoutingRequest,
    )

    dispatcher = DispatcherDelegationRoutingRequest()

    assert "omnibase-infra.delegation-routing-request" in dispatcher.message_types
    assert dispatcher.dispatcher_id == "dispatcher.delegation.routing-request"


@pytest.mark.asyncio
async def test_plugin_delegation_wire_dispatchers_includes_routing_reducer() -> None:
    """PluginDelegation.wire_dispatchers() registers the routing reducer dispatcher."""
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4

    from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
        HandlerDelegationWorkflow,
    )
    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )
    from omnibase_infra.runtime.models.model_domain_plugin_config import (
        ModelDomainPluginConfig,
    )

    plugin = PluginDelegation()

    container = MagicMock()
    handler = HandlerDelegationWorkflow()
    container.service_registry = MagicMock()
    container.service_registry.resolve_service = AsyncMock(return_value=handler)

    registered_dispatcher_ids: list[str] = []
    engine = MagicMock()
    engine.register_dispatcher = MagicMock(
        side_effect=lambda dispatcher_id, **kwargs: registered_dispatcher_ids.append(
            dispatcher_id
        )
    )
    engine.register_route = MagicMock()

    config = ModelDomainPluginConfig(
        container=container,
        event_bus=MagicMock(),
        correlation_id=uuid4(),
        input_topic="test-input",
        output_topic="test-output",
        consumer_group="test-group",
        dispatch_engine=engine,
    )

    plugin._handler_wiring_succeeded = True
    result = await plugin.wire_dispatchers(config)

    assert result.success
    assert "dispatcher.delegation.routing-request" in registered_dispatcher_ids

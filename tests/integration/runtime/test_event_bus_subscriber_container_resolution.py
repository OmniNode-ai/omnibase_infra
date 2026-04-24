# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for container-owned subscribe-capable event bus resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_publisher import (
    ProtocolEventBusPublisher,
)
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.models import ModelDomainPluginConfig

pytestmark = pytest.mark.integration


async def _register_runtime_bus(
    container: ModelONEXContainer, bus: EventBusInmemory
) -> None:
    assert container.service_registry is not None
    await container.service_registry.register_instance(ProtocolEventBusPublisher, bus)
    await container.service_registry.register_instance(ProtocolEventBusSubscriber, bus)


def _plugin_config(
    *,
    container: ModelONEXContainer,
    config_bus: EventBusInmemory,
) -> ModelDomainPluginConfig:
    return ModelDomainPluginConfig(
        container=container,
        event_bus=config_bus,
        correlation_id=uuid4(),
        input_topic="onex.cmd.test.input.v1",
        output_topic="onex.evt.test.output.v1",
        consumer_group="test-group",
        dispatch_engine=MagicMock(dispatch=AsyncMock()),
        node_identity=ModelNodeIdentity(
            env="test",
            service="omnibase_infra",
            node_name="registration-orchestrator",
            version="v1",
        ),
    )


@pytest.mark.asyncio
async def test_container_resolves_same_runtime_bus_for_publish_and_subscribe() -> None:
    """The real service registry exposes one runtime bus through both protocols."""
    container = ModelONEXContainer()
    assert container.service_registry is not None

    bus = EventBusInmemory(environment="test", group="runtime")
    await bus.start()
    try:
        await _register_runtime_bus(container, bus)

        resolved_publisher = await container.service_registry.resolve_service(
            ProtocolEventBusPublisher
        )
        resolved_subscriber = await container.service_registry.resolve_service(
            ProtocolEventBusSubscriber
        )
    finally:
        await bus.close()

    assert resolved_publisher is bus
    assert resolved_subscriber is bus
    assert resolved_publisher is resolved_subscriber


@pytest.mark.asyncio
async def test_registration_consumers_defer_subscription_wiring_to_auto_wiring() -> (
    None
):
    """ServiceRegistration no longer owns subscription wiring locally."""
    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )

    container = ModelONEXContainer()
    assert container.service_registry is not None

    resolved_bus = EventBusInmemory(environment="test", group="resolved")
    config_bus = EventBusInmemory(environment="test", group="config-fallback")
    await resolved_bus.start()
    await config_bus.start()

    await _register_runtime_bus(container, resolved_bus)
    config = _plugin_config(container=container, config_bus=config_bus)

    plugin = ServiceRegistration()
    plugin._handler_wiring_succeeded = True
    plugin._pool = MagicMock()

    try:
        result = await plugin.start_consumers(config)
    finally:
        await resolved_bus.close()
        await config_bus.close()

    assert result.success is True
    assert "auto-wiring" in result.message.lower()

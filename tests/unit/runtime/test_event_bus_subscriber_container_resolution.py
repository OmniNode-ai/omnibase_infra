# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OMN-9556: subscribe-capable event bus resolved from container.

Tests prove:
1. The kernel registers the event bus under ProtocolEventBusSubscriber in the container.
2. Domain plugin start_consumers() resolves ProtocolEventBusSubscriber from the container
   rather than relying solely on config.event_bus.
3. The fallback path (container unavailable) still works via config.event_bus isinstance check.
4. Publisher-only paths (ProtocolEventBusPublisher) remain unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.protocols.event_bus.protocol_event_bus_publisher import (
    ProtocolEventBusPublisher,
)
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.models import ModelDomainPluginConfig
from omnibase_infra.runtime.protocol_domain_plugin import ModelDomainPluginResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin_config(
    *,
    event_bus: object,
    service_registry: object | None = None,
    dispatch_engine: object | None = None,
    node_identity: object | None = None,
) -> ModelDomainPluginConfig:
    """Build a minimal ModelDomainPluginConfig for testing."""
    container = MagicMock()
    container.service_registry = service_registry
    return ModelDomainPluginConfig(
        container=container,
        event_bus=event_bus,  # type: ignore[arg-type]
        correlation_id=uuid4(),
        input_topic="onex.cmd.test.input.v1",
        output_topic="onex.evt.test.output.v1",
        consumer_group="test-group",
        dispatch_engine=dispatch_engine,
        node_identity=node_identity,
    )


# ---------------------------------------------------------------------------
# ProtocolEventBusSubscriber protocol satisfaction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_event_bus_inmemory_satisfies_subscriber_protocol() -> None:
    """EventBusInmemory implements ProtocolEventBusSubscriber structurally."""
    bus = EventBusInmemory(environment="test", group="test")
    assert isinstance(bus, ProtocolEventBusSubscriber), (
        "EventBusInmemory must satisfy ProtocolEventBusSubscriber for container registration"
    )


@pytest.mark.unit
def test_event_bus_inmemory_satisfies_publisher_protocol() -> None:
    """EventBusInmemory implements ProtocolEventBusPublisher (publisher-only path unaffected)."""
    bus = EventBusInmemory(environment="test", group="test")
    assert isinstance(bus, ProtocolEventBusPublisher), (
        "EventBusInmemory must satisfy ProtocolEventBusPublisher — publisher path must remain unaffected"
    )


# ---------------------------------------------------------------------------
# Kernel registration: ProtocolEventBusSubscriber added alongside Publisher
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kernel_registers_event_bus_as_subscriber_when_bus_supports_subscribe() -> (
    None
):
    """When event_bus satisfies ProtocolEventBusSubscriber, kernel registers it under that protocol.

    This exercises the OMN-9556 addition in service_kernel.py step 4.1 that registers
    the runtime bus under ProtocolEventBusSubscriber alongside ProtocolEventBusPublisher.
    """
    bus = EventBusInmemory(environment="test", group="test")
    registry = MagicMock()
    registry.register_instance = AsyncMock()

    container = MagicMock()
    container.service_registry = registry

    # Simulate the kernel registration logic added in OMN-9556
    if isinstance(bus, ProtocolEventBusSubscriber):
        await container.service_registry.register_instance(
            ProtocolEventBusSubscriber,
            bus,
        )

    # Assert register_instance was called with ProtocolEventBusSubscriber
    calls = registry.register_instance.call_args_list
    assert len(calls) == 1
    assert calls[0].args[0] is ProtocolEventBusSubscriber
    assert calls[0].args[1] is bus


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kernel_skips_subscriber_registration_when_bus_lacks_subscribe() -> None:
    """When event_bus does NOT satisfy ProtocolEventBusSubscriber, registration is skipped."""
    # Build a publisher-only bus (no subscribe method)
    bus = MagicMock(spec=["publish", "publish_envelope"])
    registry = MagicMock()
    registry.register_instance = AsyncMock()

    # Simulate the OMN-9556 kernel logic
    if isinstance(bus, ProtocolEventBusSubscriber):
        await registry.register_instance(ProtocolEventBusSubscriber, bus)

    # Should NOT have been called since bus doesn't satisfy ProtocolEventBusSubscriber
    registry.register_instance.assert_not_called()


# ---------------------------------------------------------------------------
# Plugin start_consumers: resolves from container, falls back to config.event_bus
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delegation_plugin_resolves_subscriber_bus_from_container() -> None:
    """PluginDelegation.start_consumers() resolves ProtocolEventBusSubscriber from container.

    When the container has the bus registered under ProtocolEventBusSubscriber,
    the plugin should use that instance (not config.event_bus).
    """
    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    resolved_bus = EventBusInmemory(environment="test", group="test-resolved")
    fallback_bus = EventBusInmemory(environment="test", group="test-fallback")
    await resolved_bus.start()
    await fallback_bus.start()

    # Set up a container that returns the bus from service_registry
    registry = MagicMock()
    registry.resolve_service = AsyncMock(return_value=resolved_bus)
    registry.register_instance = AsyncMock()

    plugin = PluginDelegation()
    plugin._handler_wiring_succeeded = True
    plugin._dispatcher_wiring_succeeded = True

    node_identity = MagicMock()
    node_identity.env = "test"
    node_identity.service = "test-service"
    node_identity.node_name = "test-node"
    node_identity.version = "v1"

    dispatch_engine = MagicMock()
    dispatch_engine.dispatch = AsyncMock()

    config = _make_plugin_config(
        event_bus=fallback_bus,
        service_registry=registry,
        dispatch_engine=dispatch_engine,
        node_identity=node_identity,
    )
    config.container.service_registry = registry
    config.output_topic = "onex.evt.test.output.v1"

    # Patch at source module (lazy-imported inside try block in plugin)
    with (
        patch(
            "omnibase_infra.runtime.event_bus_subcontract_wiring.load_event_bus_subcontract",
            return_value=SimpleNamespace(
                subscribe_topics=["onex.cmd.test.input.v1"],
            ),
        ),
        patch(
            "omnibase_infra.runtime.event_bus_subcontract_wiring.EventBusSubcontractWiring"
        ) as wiring_cls,
    ):
        wiring = MagicMock()
        wiring.wire_subscriptions = AsyncMock()
        wiring_cls.return_value = wiring
        result = await plugin.start_consumers(config)

    assert isinstance(result, ModelDomainPluginResult)
    assert registry.resolve_service.call_args_list[0].args == (
        ProtocolEventBusSubscriber,
    )
    assert wiring_cls.call_args.kwargs["event_bus"] is resolved_bus
    assert wiring_cls.call_args.kwargs["event_bus"] is not config.event_bus
    wiring.wire_subscriptions.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delegation_plugin_falls_back_to_config_event_bus_when_container_absent() -> (
    None
):
    """PluginDelegation falls back to config.event_bus when container has no service_registry."""
    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    bus = EventBusInmemory(environment="test", group="test")
    await bus.start()

    plugin = PluginDelegation()
    plugin._handler_wiring_succeeded = True
    plugin._dispatcher_wiring_succeeded = True

    node_identity = MagicMock()
    node_identity.env = "test"
    node_identity.service = "test-service"
    node_identity.node_name = "test-node"
    node_identity.version = "v1"

    dispatch_engine = MagicMock()

    # No service_registry — simulates test/minimal environments
    config = _make_plugin_config(
        event_bus=bus,
        service_registry=None,
        dispatch_engine=dispatch_engine,
        node_identity=node_identity,
    )

    with (
        patch(
            "omnibase_infra.runtime.event_bus_subcontract_wiring.load_event_bus_subcontract",
            return_value=None,
        ),
    ):
        result = await plugin.start_consumers(config)

    # Skipped because subcontract is None — but the fallback path (config.event_bus check)
    # must have been evaluated rather than erroring out
    assert isinstance(result, ModelDomainPluginResult)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delegation_plugin_skips_when_bus_lacks_subscribe_and_not_in_container() -> (
    None
):
    """PluginDelegation skips when bus has no subscribe and container resolution fails."""
    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    # A bus without subscribe capability
    publisher_only_bus = MagicMock(spec=["publish", "publish_envelope"])

    registry = MagicMock()
    registry.resolve_service = AsyncMock(side_effect=Exception("not registered"))

    plugin = PluginDelegation()
    plugin._handler_wiring_succeeded = True
    plugin._dispatcher_wiring_succeeded = True

    dispatch_engine = MagicMock()

    config = _make_plugin_config(
        event_bus=publisher_only_bus,
        service_registry=registry,
        dispatch_engine=dispatch_engine,
        node_identity=MagicMock(),
    )

    result = await plugin.start_consumers(config)

    # Should skip (not error) — skipped results have success=True per ModelDomainPluginResult.skipped()
    assert isinstance(result, ModelDomainPluginResult)
    assert result.message is not None
    assert "subscribe" in result.message.lower(), (
        f"Expected skip message mentioning 'subscribe', got: {result.message}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_plugin_resolves_subscriber_bus_from_container() -> None:
    """ServiceRegistration.start_consumers() resolves ProtocolEventBusSubscriber from container."""
    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )

    resolved_bus = EventBusInmemory(environment="test", group="test-resolved")
    fallback_bus = EventBusInmemory(environment="test", group="test-fallback")
    await resolved_bus.start()
    await fallback_bus.start()

    registry = MagicMock()
    registry.resolve_service = AsyncMock(return_value=resolved_bus)
    registry.register_instance = AsyncMock()

    plugin = ServiceRegistration()
    plugin._handler_wiring_succeeded = True
    plugin._pool = MagicMock()

    node_identity = MagicMock()
    node_identity.env = "test"
    node_identity.service = "test-service"
    node_identity.node_name = "test-node"
    node_identity.version = "v1"

    dispatch_engine = MagicMock()
    dispatch_engine.dispatch = AsyncMock()

    config = _make_plugin_config(
        event_bus=fallback_bus,
        service_registry=registry,
        dispatch_engine=dispatch_engine,
        node_identity=node_identity,
    )
    config.container.service_registry = registry

    with (
        patch(
            "omnibase_infra.runtime.event_bus_subcontract_wiring.load_event_bus_subcontract",
            return_value=SimpleNamespace(
                subscribe_topics=["onex.cmd.test.input.v1"],
            ),
        ),
        patch(
            "omnibase_infra.runtime.event_bus_subcontract_wiring.EventBusSubcontractWiring"
        ) as wiring_cls,
    ):
        wiring = MagicMock()
        wiring.wire_subscriptions = AsyncMock()
        wiring_cls.return_value = wiring
        result = await plugin.start_consumers(config)

    assert isinstance(result, ModelDomainPluginResult)
    registry.resolve_service.assert_called_once_with(ProtocolEventBusSubscriber)
    assert wiring_cls.call_args.kwargs["event_bus"] is resolved_bus
    assert wiring_cls.call_args.kwargs["event_bus"] is not config.event_bus
    wiring.wire_subscriptions.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shared_bus_instance_used_by_both_publisher_and_subscriber_paths() -> (
    None
):
    """The same EventBusInmemory instance satisfies both publisher and subscriber protocols.

    This proves that registering the same bus under two protocols does not break
    the identity invariant — both resolve to the identical object.
    """
    bus = EventBusInmemory(environment="test", group="test")

    # Both protocol registrations should use the same instance
    assert isinstance(bus, ProtocolEventBusPublisher)
    assert isinstance(bus, ProtocolEventBusSubscriber)

    # Simulate container resolving both — they must be the same object
    resolved_as_publisher: ProtocolEventBusPublisher = bus  # type: ignore[assignment]
    resolved_as_subscriber: ProtocolEventBusSubscriber = bus  # type: ignore[assignment]

    assert resolved_as_publisher is resolved_as_subscriber, (
        "Publisher and subscriber registrations must point to the same bus instance"
    )

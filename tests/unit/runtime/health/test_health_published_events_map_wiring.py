# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for published_events_map health check wiring into RuntimeHostProcess.

Verifies that health_check() includes the published_events_map component
when a DispatchResultApplier is registered in the DI container, and that
an unhealthy map causes the overall health to report unhealthy.

OMN-5164
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers


def _make_registry(
    applier: DispatchResultApplier | None = None,
) -> MagicMock:
    """Create a mock service registry that resolves DispatchResultApplier.

    Args:
        applier: The applier to return from try_resolve_service. None means
            the applier is not registered.

    Returns:
        MagicMock registry with async resolve methods.
    """
    registry = MagicMock()

    async def try_resolve(interface: type, scope: object = None) -> object | None:
        if applier is not None and interface is DispatchResultApplier:
            return applier
        return None

    async def resolve(interface: type, scope: object = None) -> object:
        result = await try_resolve(interface, scope)
        if result is None:
            raise KeyError(f"Service {interface} not registered")
        return result

    registry.try_resolve_service = AsyncMock(side_effect=try_resolve)
    registry.resolve_service = AsyncMock(side_effect=resolve)
    return registry


def _make_container_with_applier(
    output_topic_map: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock ONEX container with a DispatchResultApplier in its registry.

    Args:
        output_topic_map: The topic map to set on the applier. None means
            the applier will have an empty map (default DispatchResultApplier
            behavior converts None to {}).

    Returns:
        MagicMock container with service_registry configured.
    """
    applier = DispatchResultApplier(
        event_bus=MagicMock(),
        output_topic="fallback.topic",
        output_topic_map=output_topic_map,
    )

    container = MagicMock()
    container.service_registry = _make_registry(applier=applier)
    return container


def _make_container_without_applier() -> MagicMock:
    """Create a mock ONEX container where DispatchResultApplier is not registered."""
    container = MagicMock()
    container.service_registry = _make_registry(applier=None)
    return container


@pytest.mark.unit
class TestHealthCheckPublishedEventsMapWiring:
    """Tests for published_events_map component in health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_includes_healthy_published_events_map(
        self,
    ) -> None:
        """health_check() includes healthy published_events_map component."""
        container = _make_container_with_applier(
            output_topic_map={
                "ModelNodeRegistered": "onex.evt.platform.node-registered.v1",
            }
        )
        bus = EventBusInmemory()
        process = RuntimeHostProcess(
            container=container,
            event_bus=bus,
            config=make_runtime_config(),
        )
        seed_mock_handlers(process)
        await process.start()
        try:
            health = await process.health_check()
            components = health.get("components", [])
            assert isinstance(components, list)
            assert len(components) == 1

            component = components[0]
            assert component["name"] == "published_events_map"
            assert component["status"] == "healthy"
            assert component["details"]["entry_count"] == 1
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_map_empty(self) -> None:
        """health_check() reports unhealthy when published_events_map is empty."""
        container = _make_container_with_applier(output_topic_map={})
        bus = EventBusInmemory()
        process = RuntimeHostProcess(
            container=container,
            event_bus=bus,
            config=make_runtime_config(),
        )
        seed_mock_handlers(process)
        await process.start()
        try:
            health = await process.health_check()
            components = health.get("components", [])
            assert len(components) == 1

            component = components[0]
            assert component["name"] == "published_events_map"
            assert component["status"] == "unhealthy"
            assert "empty or not loaded" in component["error"]

            # Overall health should be False when map is unhealthy
            assert health["healthy"] is False
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_no_components_without_applier(self) -> None:
        """health_check() returns empty components when no applier is registered."""
        container = _make_container_without_applier()
        bus = EventBusInmemory()
        process = RuntimeHostProcess(
            container=container,
            event_bus=bus,
            config=make_runtime_config(),
        )
        seed_mock_handlers(process)
        await process.start()
        try:
            health = await process.health_check()
            components = health.get("components", [])
            assert components == []
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_no_components_without_container(self) -> None:
        """health_check() returns empty components when no container is provided."""
        bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=bus,
            config=make_runtime_config(),
        )
        seed_mock_handlers(process)
        await process.start()
        try:
            health = await process.health_check()
            components = health.get("components", [])
            assert components == []
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_dispatch_result_applier_published_events_map_property(
        self,
    ) -> None:
        """DispatchResultApplier.published_events_map exposes the output_topic_map."""
        topic_map = {
            "ModelNodeRegistered": "onex.evt.platform.node-registered.v1",
            "ModelNodeBecameActive": "onex.evt.platform.node-became-active.v1",
        }
        applier = DispatchResultApplier(
            event_bus=MagicMock(),
            output_topic="fallback.topic",
            output_topic_map=topic_map,
        )
        assert applier.published_events_map == topic_map

    @pytest.mark.asyncio
    async def test_dispatch_result_applier_empty_map_by_default(self) -> None:
        """DispatchResultApplier.published_events_map returns empty dict by default."""
        applier = DispatchResultApplier(
            event_bus=MagicMock(),
            output_topic="fallback.topic",
        )
        assert applier.published_events_map == {}

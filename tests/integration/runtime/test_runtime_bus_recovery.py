# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recovery integrates with real handler wiring and runtime HTTP health probes."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientSession

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka, EventBusRecovering
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.model_event_bus_recovery_config import (
    ModelEventBusRecoveryConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.health_checker import ServiceHealth
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.integration


def make_bus() -> EventBusRecovering:
    local_bus = EventBusInmemory(max_history=10)
    return EventBusRecovering(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092"),
        recovery=ModelEventBusRecoveryConfig.from_contract(),
        local_bus=local_bus,
        local_start=local_bus.start,
        local_close=local_bus.close,
        local_health_check=local_bus.health_check,
        alarm_topic="runtime-bus",
        consumer_group="test-runtime",
    )


async def test_real_contract_manifest_wires_during_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.auto_wiring.discovery import (
        discover_contracts_from_paths,
    )
    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    contract = (
        Path(__file__).parents[3]
        / "src/omnibase_infra/nodes/node_impact_analyzer_compute/contract.yaml"
    )
    manifest = discover_contracts_from_paths([contract])
    assert manifest.total_discovered == 1
    bus = make_bus()
    monkeypatch.setattr(
        EventBusKafka, "start", AsyncMock(side_effect=InfraConnectionError("down"))
    )
    try:
        await bus.start()
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=MessageDispatchEngine(),
            event_bus=bus,
            environment="test",
            container=ModelONEXContainer(),
            subscribe_immediately=False,
            result_appliers_by_contract={},
            materialized_explicit_dependencies={},
            topology=None,
        )
        assert report.total_failed == 0
        assert report.total_wired == 1
    finally:
        await bus.close()


async def test_http_probes_track_recovery_without_process_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same HTTP server stays live while shared-bus readiness recovers."""
    bus = make_bus()
    retry = asyncio.Event()
    attempts = 0

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(EventBusKafka, "subscribe", AsyncMock(return_value=AsyncMock()))
    monkeypatch.setattr(bus, "_sleep_before_retry", lambda delay: retry.wait())
    process = RuntimeHostProcess(
        container=MagicMock(service_registry=None),
        event_bus=bus,
        config=make_runtime_config(),
    )
    seed_mock_handlers(process)
    for method in (
        "_discover_or_wire_handlers",
        "_populate_handlers_from_registry",
        "_wire_package_node_subscriptions",
    ):
        monkeypatch.setattr(process, method, AsyncMock())
    server = ServiceHealth(runtime=process, host="127.0.0.1", port=0)
    try:
        await process.start()
        await server.start()
        address = server._runner.addresses[0]
        url = f"http://127.0.0.1:{address[1]}"
        async with ClientSession() as client:
            async with client.get(f"{url}/health") as response:
                assert response.status == 200
                assert (await response.json())["status"] == "degraded"
            async with client.get(f"{url}/ready") as response:
                assert response.status == 503
            assert len(bus.consumer_rejoin_alarms()) == 1
            retry.set()
            async with asyncio.timeout(2):
                while not (await process.readiness_check())["ready"]:
                    await asyncio.sleep(0)
            async with client.get(f"{url}/ready") as response:
                assert response.status == 200
            async with client.get(f"{url}/health") as response:
                assert response.status == 200
                assert (await response.json())["status"] == "healthy"
            assert process.is_running
            assert attempts == 2
            assert len(bus.consumer_rejoin_alarms()) == 1
    finally:
        await server.stop()
        await process.stop()

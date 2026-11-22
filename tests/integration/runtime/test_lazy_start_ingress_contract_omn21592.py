# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21592 -- the lazy-start front door's real contract wires through auto-wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

NODE_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_lazy_start_ingress/contract.yaml"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingress_contract_wires_from_disk_with_zero_failures() -> None:
    manifest = discover_contracts_from_paths([NODE_CONTRACT])
    assert [c.name for c in manifest.contracts] == ["node_lazy_start_ingress"]

    bus = EventBusInmemory()
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, bus)
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        environment="local",
        subscribe_immediately=False,
        container=container,
        event_bus=bus,
    )

    assert report.total_wired == 1
    assert report.total_failed == 0

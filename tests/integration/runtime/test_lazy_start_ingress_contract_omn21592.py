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
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    boot_manifest,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

NODE_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_lazy_start_ingress/contract.yaml"
)


DORMANT_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_validation_executor/contract.yaml"
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


class _GatedBus(EventBusKafka):
    """Kafka-typed stand-in: no transport; the ingress puts its wake in front of publish."""

    def __init__(self) -> None:
        self._started = True
        self._environment = "local"
        self.subscribed: list[str] = []
        self.published: list[str] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append(topic)

    async def subscribe(self, *, topic: str, **_: object) -> object:
        self.subscribed.append(topic)

        async def _stop() -> None:
            self.subscribed.remove(topic)

        return _stop

    def get_consumer_groups(self) -> dict[tuple[str, str], str]:
        return {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingress_wiring_installs_the_wake_on_the_runtime_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real ingress contract wires with a lazy-start lane; a publish wakes a node.

    The dormant node is a real contract of this repository, wired by the real
    runtime start transaction; the publish goes through the bus's own publish path.
    """
    root = tmp_path / "overlays" / "runtime_contract_hotload"
    root.mkdir(parents=True)
    (root / "overlay.yaml").write_text(
        "schema_version: 1\nlanes:\n  dogfood:\n    enabled: true\n"
        "    mode: on_demand\n    allowed_environments: [local]\n"
        "    lazy_start:\n      ingress_contract: node_lazy_start_ingress\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(tmp_path / "overlays"))
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dogfood")
    # The lazy wake now judges the runtime bus itself (OMN-21753), so the version
    # handover (OMN-19998) runs as it does for a contract wired at boot; this
    # stand-in has no transport to list groups on.
    monkeypatch.setenv("ONEX_RUNTIME_VERSION_HANDOVER", '{"enabled": false}')
    StoreLazyStartDormant.clear()

    discovered = discover_contracts_from_paths([NODE_CONTRACT, DORMANT_CONTRACT])
    dormant = discovered.contracts[1]
    assert dormant.event_bus is not None and dormant.event_bus.subscribe_topics
    boot = boot_manifest(discovered)
    assert [c.name for c in boot.contracts] == ["node_lazy_start_ingress"]

    bus = _GatedBus()
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, bus)
    report = await wire_from_manifest(
        boot,
        MessageDispatchEngine(),
        environment="local",
        subscribe_immediately=False,
        container=container,
        event_bus=bus,
    )
    assert report.total_wired == 1
    assert report.total_failed == 0
    assert bus.subscribed == []

    topic = dormant.event_bus.subscribe_topics[0]
    await bus.publish(topic, None, b"{}")

    assert bus.published == [topic]
    assert topic in bus.subscribed
    StoreLazyStartDormant.clear()

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pattern B broker integration against a real omnimarket handler."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from omnimarket.nodes.node_aislop_sweep.handlers.handler_aislop_sweep import (
    AislopSweepRequest,
    NodeAislopSweep,
)

from omnibase_core.dispatch.dispatch_bus_client import DispatchBusClient
from omnibase_core.models.dispatch.model_dispatch_bus_route import (
    ModelDispatchBusRoute,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.generated import EnumOmnibaseInfraTopic
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.runtime_local_ingress import RuntimeLocalIngressRoute
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

pytestmark = pytest.mark.integration


def _route() -> RuntimeLocalIngressRoute:
    package = importlib.import_module("omnimarket")
    node_dir = Path(package.__file__).resolve().parent
    contract_path = node_dir / "nodes" / "node_aislop_sweep" / "contract.yaml"
    return RuntimeLocalIngressRoute(
        node_name="node_aislop_sweep",
        contract_name="aislop_sweep",
        command_topic="onex.cmd.omnimarket.aislop-sweep-start.v1",
        event_type="omnimarket.aislop-sweep-start",
        terminal_event="onex.evt.omnimarket.aislop-sweep-completed.v1",
        contract_path=str(contract_path),
        package_name="omnimarket",
    )


@pytest.mark.asyncio
async def test_pattern_b_broker_dispatches_real_aislop_handler(tmp_path: Path) -> None:
    route = _route()
    repo_dir = tmp_path / "demo_repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "demo.py").write_text(
        'ONEX_EVENT_BUS_TYPE = "inmemory"\n',
        encoding="utf-8",
    )

    bus = EventBusInmemory(environment="test", group="pattern-b-omnimarket")
    await bus.start()

    broker = RuntimePatternBBroker(
        bus,
        command_topic=EnumOmnibaseInfraTopic.CMD_PATTERN_B_DISPATCH_V1.value,
        routes={"aislop_sweep": route},
    )
    await broker.start()

    handler = NodeAislopSweep(event_bus=bus)

    async def worker(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        request = AislopSweepRequest.model_validate(envelope.payload)
        result = handler.handle(request)
        terminal_envelope = ModelEventEnvelope[object](
            payload=result.model_dump(mode="json"),
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_event or "unknown",
            source_tool="aislop_sweep",
        )
        await bus.publish(
            route.terminal_event or "unknown",
            None,
            terminal_envelope.model_dump_json().encode("utf-8"),
            None,
        )

    await bus.subscribe(
        route.command_topic, group_id="aislop-worker", on_message=worker
    )

    client = DispatchBusClient(bus, source="pytest")
    broker_route = ModelDispatchBusRoute(
        contract_path=Path(route.contract_path),
        command_topic=EnumOmnibaseInfraTopic.CMD_PATTERN_B_DISPATCH_V1.value,
        terminal_topic="onex.evt.omnibase-infra.pattern-b-dispatch-test-completed.v1",
    )

    try:
        result = await client.request(
            broker_route,
            command_name="aislop_sweep",
            payload={"target_dirs": [str(repo_dir)], "dry_run": True},
            timeout_seconds=10,
        )
    finally:
        await broker.stop()
        await bus.close()

    assert result.status == "completed"
    assert isinstance(result.payload, dict)
    findings = result.payload.get("findings")
    assert isinstance(findings, list)
    assert findings
    first = findings[0]
    assert isinstance(first, dict)
    assert first["check"] == "prohibited-patterns"
    assert first["repo"] == "demo_repo"

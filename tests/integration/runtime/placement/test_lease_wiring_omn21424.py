# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lease nodes boot through the kernel's own wiring and answer on the bus (OMN-21424).

Loads the three real lease contracts from disk, runs ``wire_from_manifest`` over a
real ``MessageDispatchEngine`` and ``EventBusInmemory``, as the kernel does, with
the orchestrator's contract-declared configuration pointing at a throwaway etcd.
Commands published on the command topics come back as events on the event topics.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.integration.runtime.placement.conftest import EtcdProcess

pytestmark = [pytest.mark.integration]

_NODES = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"
_LEASE_NODES = (
    "node_lease_orchestrator",
    "node_lease_reducer",
    "node_projection_lease",
)

ACQUIRE = "onex.cmd.omnibase-infra.lease-acquire-requested.v1"
RENEW = "onex.cmd.omnibase-infra.lease-renew-requested.v1"
RELEASE = "onex.cmd.omnibase-infra.lease-release-requested.v1"
EVENTS = (
    "onex.evt.omnibase-infra.lease-granted.v1",
    "onex.evt.omnibase-infra.lease-refused.v1",
    "onex.evt.omnibase-infra.lease-renewed.v1",
    "onex.evt.omnibase-infra.lease-released.v1",
    "onex.evt.omnibase-infra.lease-expired.v1",
)


class _Wired:
    def __init__(self, bus: EventBusInmemory) -> None:
        self.bus = bus
        self.seen: dict[str, list[dict[str, object]]] = {t: [] for t in EVENTS}

    async def send(self, topic: str, payload: dict[str, object]) -> None:
        envelope = ModelEventEnvelope[object](
            payload=payload, correlation_id=uuid4(), event_type=topic
        )
        await self.bus.publish(topic, None, envelope.model_dump_json().encode(), None)

    async def next_on(self, topic: str, count: int) -> dict[str, object]:
        async def until() -> None:
            while len(self.seen[topic]) < count:
                await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(until(), timeout=10)
        except TimeoutError:
            pytest.fail(
                f"no event #{count} on {topic}; seen: "
                + json.dumps({t: len(v) for t, v in self.seen.items()})
            )
        return self.seen[topic][count - 1]


@pytest.fixture
async def wired(
    etcd: EtcdProcess, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Wired]:
    monkeypatch.setenv("ONEX_LEASE_ETCD_ENDPOINT", etcd.endpoint)
    monkeypatch.setenv("ONEX_LEASE_KINDS", "exclusive:1:30")
    manifest = discover_contracts_from_paths(
        [_NODES / node / "contract.yaml" for node in _LEASE_NODES]
    )
    assert manifest.errors == ()
    bus = EventBusInmemory(environment="test", group="omn-21424-wiring")
    await bus.start()
    state = _Wired(bus)
    try:

        def collector(topic: str) -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def collect(message: ModelEventMessage) -> None:
                envelope = ModelEventEnvelope[object].model_validate_json(message.value)
                assert isinstance(envelope.payload, dict)
                state.seen[topic].append(envelope.payload)

            return collect

        for topic in EVENTS:
            await bus.subscribe(
                topic,
                group_id=f"omn-21424-collect-{topic}",
                on_message=collector(topic),
            )
        engine = MessageDispatchEngine()
        report = await wire_from_manifest(
            manifest, engine, event_bus=bus, environment="local"
        )
        engine.freeze()
        assert report.quarantined_handlers == (), report
        wired_handlers = {
            handler.handler_name
            for contract in report.results
            for handler in contract.wirings
        }
        assert {
            "HandlerLeaseAcquire",
            "HandlerLeaseRenew",
            "HandlerLeaseRelease",
            "HandlerLeaseReducer",
        } <= wired_handlers, report
        yield state
    finally:
        await bus.close()


async def test_acquire_on_the_command_topic_is_granted_once_then_refused(
    wired: _Wired,
) -> None:
    first = {
        "kind": "exclusive",
        "resource": "pr:omn-21424-wiring",
        "holder": "lane-a",
        "request_id": str(uuid4()),
    }
    await wired.send(ACQUIRE, first)
    granted = await wired.next_on("onex.evt.omnibase-infra.lease-granted.v1", 1)
    assert granted["holder"] == "lane-a"
    assert int(str(granted["fence"])) > 0

    await wired.send(ACQUIRE, {**first, "holder": "lane-b", "request_id": str(uuid4())})
    refused = await wired.next_on("onex.evt.omnibase-infra.lease-refused.v1", 1)
    assert refused["holder"] == "lane-b"
    assert int(str(refused["fence"])) == 0

    await wired.send(ACQUIRE, first)
    replayed = await wired.next_on("onex.evt.omnibase-infra.lease-granted.v1", 2)
    assert replayed["lease_id"] == granted["lease_id"]
    assert replayed["fence"] == granted["fence"]


async def test_renew_and_release_answer_on_their_own_topics(wired: _Wired) -> None:
    await wired.send(
        ACQUIRE,
        {
            "kind": "exclusive",
            "resource": "slot:omn-21424-wiring",
            "holder": "lane-c",
            "request_id": str(uuid4()),
        },
    )
    granted = await wired.next_on("onex.evt.omnibase-infra.lease-granted.v1", 1)
    lease = {
        "kind": "exclusive",
        "resource": "slot:omn-21424-wiring",
        "lease_id": str(UUID(str(granted["lease_id"]))),
    }
    await wired.send(RENEW, lease)
    renewed = await wired.next_on("onex.evt.omnibase-infra.lease-renewed.v1", 1)
    assert renewed["fence"] == granted["fence"]

    await wired.send(RELEASE, lease)
    released = await wired.next_on("onex.evt.omnibase-infra.lease-released.v1", 1)
    assert released["fence"] == granted["fence"]

    await wired.send(RENEW, lease)
    expired = await wired.next_on("onex.evt.omnibase-infra.lease-expired.v1", 1)
    assert expired["lease_id"] == lease["lease_id"]

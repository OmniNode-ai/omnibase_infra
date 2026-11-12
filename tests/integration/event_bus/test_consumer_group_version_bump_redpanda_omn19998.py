# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: a contract version bump loses no rows, on a real broker.

``tests/unit/runtime/auto_wiring/test_consumer_group_version_bump_handover.py``
runs the scenario on the in-process group-offset substrate. This file runs it on
Redpanda, where the group is per-topic scoped (``.__t.<topic>``), the topic has
three partitions, and offsets are read and committed by the broker's group
coordinator:

1. a runtime wires the projection at contract version 1.1.1 and projects the
   events published to it; the runtime then stops (its bus closes);
2. events are published to all three partitions while nothing is joined (the
   deploy gap; partition 2 was never committed by 1.1.1);
3. a second runtime wires the same projection at 1.2.0 through
   ``wire_from_manifest``; more events follow.

Every event must be projected exactly once across the two versions. With the
version handover switched off the gap is lost, which is the defect.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import Counter
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    list_groups_on_kafka,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.event_bus.model_consumer_group_fence import (
    ModelConsumerGroupFence,
)
from omnibase_infra.models.event_bus.model_version_handover_policy import (
    VERSION_HANDOVER_ENV_VAR,
    select_version_predecessors,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import ModelHandlerRef
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.utils import compute_consumer_group_id
from tests.helpers.service_env import require_service_env

from .conftest import BUS_TEST_ENVIRONMENT

pytestmark = [pytest.mark.integration, pytest.mark.kafka]

PARTITIONS = 3
NODE = "projection_work_ledger"

_projected: list[str] = []


class _ProjectionWriter:
    async def handle(self, request: dict[str, Any]) -> None:
        _projected.append(str(request["payload"]["event_id"]))


@pytest.fixture(autouse=True, scope="module")
def _require_kafka() -> None:
    require_service_env(
        opt_in="KAFKA_INTEGRATION_TESTS",
        endpoint="KAFKA_BOOTSTRAP_SERVERS",
        workflow=".github/workflows/ci.yml (service-integration-suites)",
        service="Kafka/Redpanda",
    )


@pytest.fixture
def bootstrap() -> str:
    return os.environ["KAFKA_BOOTSTRAP_SERVERS"]


@pytest.fixture
async def topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
) -> str:
    return await ensure_test_topic(
        f"onex.evt.omnimarket.omn19998-bump-{uuid.uuid4().hex[:10]}.v1",
        PARTITIONS,
    )


@pytest.fixture
async def new_bus(bootstrap: str) -> AsyncGenerator[Callable[[], Any], None]:
    """Hand out started buses; every one still open is closed afterwards."""
    opened: list[EventBusKafka] = []

    async def make() -> EventBusKafka:
        kafka = EventBusKafka(
            config=ModelKafkaEventBusConfig(
                bootstrap_servers=bootstrap,
                environment=BUS_TEST_ENVIRONMENT,
                timeout_seconds=30,
                max_retry_attempts=2,
                retry_backoff_base=0.5,
            )
        )
        await kafka.start()
        opened.append(kafka)
        return kafka

    yield make
    for kafka in opened:
        await kafka.close()


def _contract(
    tmp_path: Path, topic: str, version: tuple[int, int, int]
) -> ModelDiscoveredContract:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        f"name: {NODE}\n"
        "node_type: REDUCER_GENERIC\n"
        f"contract_version: {'.'.join(map(str, version))}\n"
        "description: OMN-19998 projection fixture\n"
        f"event_bus:\n  subscribe_topics:\n    - {topic}\n",
        encoding="utf-8",
    )
    major, minor, patch_ = version
    return ModelDiscoveredContract(
        name=NODE,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=major, minor=minor, patch=patch_),
        contract_path=contract_path,
        entry_point_name=NODE,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="_ProjectionWriter", module=__name__),
                    event_model=None,
                ),
            ),
        ),
    )


async def _publish(bootstrap: str, topic: str, events: list[tuple[int, str]]) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        for partition, event_id in events:
            envelope = {
                "event_type": topic,
                "payload": {"event_id": event_id},
                "correlation_id": str(uuid.uuid4()),
            }
            await producer.send_and_wait(
                topic,
                json.dumps(envelope).encode(),
                key=event_id.encode(),
                partition=partition,
            )
    finally:
        await producer.stop()


async def _wire(
    tmp_path: Path,
    topic: str,
    bus: EventBusKafka,
    environment: str,
    version: tuple[int, int, int],
) -> None:
    engine = MessageDispatchEngine()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_ProjectionWriter,
    ):
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(_contract(tmp_path, topic, version),)),
            engine,
            event_bus=bus,
            environment=environment,
        )
    engine.freeze()


async def _await_projected(expected: set[str]) -> None:
    async with asyncio.timeout(60):
        while not expected <= set(_projected):
            await asyncio.sleep(0.2)


async def _run_the_bump(
    tmp_path: Path,
    bootstrap: str,
    topic: str,
    new_bus: Callable[[], Any],
) -> tuple[list[str], list[str], str]:
    """1.1.1 consumes and its runtime stops, the gap passes, 1.2.0 joins."""
    _projected.clear()
    environment = f"omn19998-{uuid.uuid4().hex[:8]}"

    old_bus = await new_bus()
    await _wire(tmp_path, topic, old_bus, environment, (1, 1, 1))
    # The first join must be settled before the first event, or ``latest``
    # would skip it on this test's own account.
    await asyncio.sleep(5.0)
    before = [(0, "a-0"), (0, "a-1"), (1, "a-2")]
    await _publish(bootstrap, topic, before)
    await _await_projected({"a-0", "a-1", "a-2"})
    await old_bus.close()  # the rebuild stops the old runtime
    old = list(_projected)

    gap = [(0, "gap-0"), (1, "gap-1"), (2, "gap-2")]
    await _publish(bootstrap, topic, gap)

    successor_bus = await new_bus()
    await _wire(tmp_path, topic, successor_bus, environment, (1, 2, 0))
    after = [(0, "after-0"), (2, "after-1")]
    await _publish(bootstrap, topic, after)
    await _await_projected({"after-0", "after-1"})
    await asyncio.sleep(3.0)  # a duplicate would arrive here
    return old, list(_projected)[len(old) :], environment


@pytest.mark.asyncio
async def test_version_bump_loses_nothing_on_redpanda(
    tmp_path: Path,
    bootstrap: str,
    topic: str,
    new_bus: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC1 on the broker: before, gap and after, each exactly once."""
    monkeypatch.delenv(VERSION_HANDOVER_ENV_VAR, raising=False)
    old, new, environment = await _run_the_bump(tmp_path, bootstrap, topic, new_bus)

    assert sorted(old) == ["a-0", "a-1", "a-2"]
    assert sorted(new) == sorted(["gap-0", "gap-1", "gap-2", "after-0", "after-1"])
    every = Counter(old + new)
    assert set(every.values()) == {1}, every

    # The seam: the listing the handover reads names the predecessor the
    # selection rule picks for the successor, on the broker's own group ids.
    bus = await new_bus()
    listed = await list_groups_on_kafka(bus)
    successor = compute_consumer_group_id(
        ModelNodeIdentity(
            env=environment, service="omnimarket", node_name=NODE, version="1.2.0"
        )
    )
    predecessors = select_version_predecessors(f"{successor}.__t.{topic}", listed)
    assert len(predecessors) == 1
    assert ".consume.1.1.1" in predecessors[0]


@pytest.mark.asyncio
async def test_without_version_handover_the_gap_is_lost_on_redpanda(
    tmp_path: Path,
    bootstrap: str,
    topic: str,
    new_bus: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2 on the broker: the defect. A new group at ``latest`` skips the gap."""
    monkeypatch.setenv(VERSION_HANDOVER_ENV_VAR, '{"enabled": false}')
    old, new, _ = await _run_the_bump(tmp_path, bootstrap, topic, new_bus)

    assert sorted(old) == ["a-0", "a-1", "a-2"]
    assert sorted(new) == ["after-0", "after-1"]

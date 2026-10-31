# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21753 (under OMN-21288): the folded-writer handover on a real broker.

The unit file runs the scenario on the in-process group-offset substrate. This
file runs it on Redpanda, where the runtime's group is per-topic scoped
(``.__t.<topic>``), the topic has three partitions, and the offsets are read
and committed by the broker's group coordinator:

1. a standalone consumer under the retired writer's group id consumes and
   commits the events on partitions 0 and 1, then leaves (the container is
   retired);
2. events are published to all three partitions while nobody is joined
   (partition 2 was never committed by the retired group, so the handover
   applies its ``earliest`` rule there);
3. the writer contract is wired through ``wire_from_manifest`` on a real
   ``EventBusKafka``; more events follow.

Every event must be projected exactly once across the container and the
runtime. Without the declaration the window is lost, which is the defect.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import Counter
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    hand_over_committed_offsets,
)
from omnibase_infra.models import ModelNodeIdentity
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
DECLARATION = """
consumer_group_handover:
  from_service: omnimarket-projections
  from_node: live-events-writer
  from_version: v1
  from_auto_offset_reset: earliest
"""

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
        f"onex.evt.omnimarket.omn21753-handover-{uuid.uuid4().hex[:10]}.v1",
        PARTITIONS,
    )


@pytest.fixture
async def bus(bootstrap: str) -> AsyncGenerator[EventBusKafka, None]:
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
    yield kafka
    await kafka.close()


def _contract(
    tmp_path: Path, topic: str, *, declaration: str
) -> ModelDiscoveredContract:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "name: projection_live_events\n"
        "node_type: REDUCER_GENERIC\n"
        "contract_version: 1.3.0\n"
        "description: OMN-21753 folded-writer fixture\n"
        f"event_bus:\n  subscribe_topics:\n    - {topic}\n" + declaration,
        encoding="utf-8",
    )
    return ModelDiscoveredContract(
        name="projection_live_events",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=3, patch=0),
        contract_path=contract_path,
        entry_point_name="projection_live_events",
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


async def _retired_container(
    bootstrap: str, topic: str, group_id: str, expected: int
) -> list[str]:
    """The standalone writer: consume, commit after processing, leave."""
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    seen: list[str] = []
    try:
        async with asyncio.timeout(60):
            while len(seen) < expected:
                batches = await consumer.getmany(timeout_ms=500)
                for partition, records in batches.items():
                    for record in records:
                        seen.append(json.loads(record.value)["payload"]["event_id"])
                    await consumer.commit({partition: records[-1].offset + 1})
    finally:
        await consumer.stop()
    return seen


async def _run_the_fold(
    tmp_path: Path, bootstrap: str, topic: str, bus: EventBusKafka, declaration: str
) -> tuple[list[str], list[str], str]:
    _projected.clear()
    environment = f"omn21753-{uuid.uuid4().hex[:8]}"
    retired_group = (
        f"{environment}.omnimarket-projections.live-events-writer.consume.v1"
    )

    before = [(0, "before-0"), (0, "before-1"), (1, "before-2")]
    await _publish(bootstrap, topic, before)
    container = await _retired_container(bootstrap, topic, retired_group, len(before))

    window = [(0, "window-0"), (1, "window-1"), (2, "window-2")]
    await _publish(bootstrap, topic, window)

    engine = MessageDispatchEngine()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_ProjectionWriter,
    ):
        await wire_from_manifest(
            ModelAutoWiringManifest(
                contracts=(_contract(tmp_path, topic, declaration=declaration),)
            ),
            engine,
            event_bus=bus,
            environment=environment,
        )
    engine.freeze()

    after = [(0, "after-0"), (2, "after-1")]
    await _publish(bootstrap, topic, after)

    async with asyncio.timeout(60):
        while "after-0" not in _projected or "after-1" not in _projected:
            await asyncio.sleep(0.2)
    await asyncio.sleep(3.0)  # a duplicate would arrive here
    return container, list(_projected), environment


@pytest.mark.asyncio
async def test_window_events_are_projected_exactly_once_on_redpanda(
    tmp_path: Path, bootstrap: str, topic: str, bus: EventBusKafka
) -> None:
    """AC1 on the broker: before, window and after, each exactly once."""
    container, runtime, environment = await _run_the_fold(
        tmp_path, bootstrap, topic, bus, DECLARATION
    )

    assert sorted(container) == ["before-0", "before-1", "before-2"]
    assert sorted(runtime) == sorted(
        ["window-0", "window-1", "window-2", "after-0", "after-1"]
    )
    every = Counter(container + runtime)
    assert set(every.values()) == {1}, every
    assert len(every) == 8

    # AC4: a second handover into the same group keeps its own positions.
    retired_group = (
        f"{environment}.omnimarket-projections.live-events-writer.consume.v1"
    )
    group = compute_consumer_group_id(
        ModelNodeIdentity(
            env=environment,
            service="omnimarket",
            node_name="projection_live_events",
            version="1.3.0",
        )
    )
    second = await hand_over_committed_offsets(
        bus,
        topic=topic,
        from_group=retired_group,
        to_group=group,
        from_auto_offset_reset="earliest",
    )
    assert second.committed == {}
    assert second.kept == (0, 1, 2)
    assert second.to_group.endswith(f".__t.{topic}")


@pytest.mark.asyncio
async def test_without_declaration_the_window_is_lost_on_redpanda(
    tmp_path: Path, bootstrap: str, topic: str, bus: EventBusKafka
) -> None:
    """AC2 on the broker: the defect. A new group at ``latest`` skips the window."""
    container, runtime, _ = await _run_the_fold(tmp_path, bootstrap, topic, bus, "")

    assert sorted(container) == ["before-0", "before-1", "before-2"]
    assert sorted(runtime) == ["after-0", "after-1"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21629: the demand watch and the health monitor agree on a deferred projection.

Two real components run together: ``HandlerKafkaTopicDemandSource`` (the boot
subscription demand split, OMN-18087) and ``ServiceRuntimeHealthMonitor``
(``run_once``). Only the Kafka client edge is replaced, at the ``aiokafka``
constructor boundary, so what is under test is the contract between the two:
the monitor's ``projection_attachment`` dimension follows the watch.

A projection whose topic holds no record is deferred to that topic's first
record, so it is absent from the live bus registry by design and must not read
as unattached. Once a record has started it, or the watch has ended for any
reason, the live registry decides again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest
from aiokafka.structs import TopicPartition

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers import handler_kafka_topic_demand_source as module
from omnibase_infra.handlers import kafka_idle_topic_watches
from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
    HandlerKafkaTopicDemandSource,
)
from omnibase_infra.handlers.kafka_idle_topic_watches import idle_watched_topics
from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    ServiceRuntimeHealthMonitor,
)

PROJECTION_TOPIC = "onex.evt.omniclaude.tool-executed.v1"
ORCHESTRATOR_TOPIC = "onex.cmd.omnimarket.contract-sweep-start.v1"
_MONITOR = "omnibase_infra.services.service_runtime_health_monitor"


class _Consumer:
    """A group-less consumer over an empty topic; ``arrivals`` delivers a record."""

    def __init__(self, offsets: dict[str, tuple[int, int]]) -> None:
        self._offsets = offsets
        self.arrivals: asyncio.Queue[str] = asyncio.Queue()
        self.getmany_error: Exception | None = None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def beginning_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {tp: self._offsets[tp.topic][0] for tp in partitions}

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {tp: self._offsets[tp.topic][1] for tp in partitions}

    def assign(self, partitions: list[TopicPartition]) -> None:
        return None

    def seek(self, tp: TopicPartition, offset: int) -> None:
        return None

    def pause(self, *partitions: TopicPartition) -> None:
        return None

    async def getmany(self, timeout_ms: int = 0) -> dict[TopicPartition, list[object]]:
        if self.getmany_error is not None:
            raise self.getmany_error
        try:
            topic = await asyncio.wait_for(self.arrivals.get(), timeout_ms / 1000)
        except TimeoutError:
            return {}
        return {TopicPartition(topic, 0): [object()]}


class _Admin:
    def __init__(self, offsets: dict[str, tuple[int, int]]) -> None:
        self._offsets = offsets

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def list_topics(self) -> list[str]:
        return sorted(self._offsets)

    async def describe_topics(self, topics: list[str]) -> list[dict[str, object]]:
        return [
            {"topic": topic, "error_code": 0, "partitions": [{"partition": 0}]}
            for topic in topics
        ]


@pytest.fixture(autouse=True)
def _isolated_idle_registry() -> Iterator[None]:
    kafka_idle_topic_watches._WATCHES.clear()
    yield
    kafka_idle_topic_watches._WATCHES.clear()


@pytest.fixture
def consumer(monkeypatch: pytest.MonkeyPatch) -> _Consumer:
    """The broker holds the projection's topic with no retained record."""
    offsets = {PROJECTION_TOPIC: (3, 3)}
    fake = _Consumer(offsets)
    monkeypatch.setattr(kafka_transport, "AIOKafkaConsumer", lambda **_: fake)
    monkeypatch.setattr(module, "AIOKafkaAdminClient", lambda **_: _Admin(offsets))
    return fake


def _projection() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_projection_session_replay",
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name="node_projection_session_replay",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(PROJECTION_TOPIC,), publish_topics=()
        ),
        db_io=ModelDbOwnershipSubcontract(
            db_tables=[
                ModelDbTableDeclaration(
                    name="session_replay",
                    database_ref="application",
                    schema="omninode_internal",
                    migration="0001_init.sql",
                    access="read_write",
                    role="projection_target",
                )
            ]
        ),
    )


def _orchestrator() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_contract_sweep",
        node_type="ORCHESTRATOR",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name="node_contract_sweep",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(ORCHESTRATOR_TOPIC,), publish_topics=()
        ),
    )


async def _health() -> ModelRuntimeHealthCheckEvent:
    """One health check; the live bus registry holds only the orchestrator."""
    manifest = ModelAutoWiringManifest(
        contracts=(_projection(), _orchestrator()), errors=()
    )
    live_group = f"runtime.sweep.consume.v1.__t.{ORCHESTRATOR_TOPIC}"
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.get_consumer_groups.return_value = {
        (ORCHESTRATOR_TOPIC, "runtime.sweep.consume.v1"): live_group
    }
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=bus, bootstrap_servers="localhost:9092"
    )
    with (
        patch(f"{_MONITOR}._discover_contracts", return_value=manifest),
        patch(
            f"{_MONITOR}._list_consumer_group_snapshots",
            return_value=[ConsumerGroupSnapshot(group_id=live_group, state="STABLE")],
        ),
    ):
        return await monitor.run_once()


def _attachment(event: ModelRuntimeHealthCheckEvent) -> str:
    return next(d.status for d in event.dimensions if d.name == "projection_attachment")


async def _until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.01)


def _source() -> HandlerKafkaTopicDemandSource:
    return HandlerKafkaTopicDemandSource(
        ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    )


async def test_a_deferred_projection_is_healthy_until_its_first_record(
    consumer: _Consumer,
) -> None:
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    _in_use, stop = await _source().watch_idle_topics({PROJECTION_TOPIC}, on_first)
    try:
        event = await _health()
        assert event.unattached_projection_count == 0
        assert _attachment(event) == "HEALTHY"
        assert started == []
    finally:
        await stop()


async def test_a_started_projection_is_judged_by_the_live_registry_again(
    consumer: _Consumer,
) -> None:
    """The start ran but registered nothing: the monitor must now report it."""
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    _in_use, stop = await _source().watch_idle_topics({PROJECTION_TOPIC}, on_first)
    try:
        await consumer.arrivals.put(PROJECTION_TOPIC)
        await _until(lambda: not idle_watched_topics())
        assert started == [PROJECTION_TOPIC]
        event = await _health()
        assert event.unattached_projection_count == 1
        assert _attachment(event) == "DEGRADED"
    finally:
        await stop()


async def test_a_stopped_watch_no_longer_masks_its_projection(
    consumer: _Consumer,
) -> None:
    async def on_first(topic: str) -> None:
        raise AssertionError("no record arrived")

    _in_use, stop = await _source().watch_idle_topics({PROJECTION_TOPIC}, on_first)
    assert _attachment(await _health()) == "HEALTHY"
    await stop()

    event = await _health()
    assert event.unattached_projection_count == 1
    assert _attachment(event) == "DEGRADED"


async def test_a_failed_watch_starts_the_projection_and_stops_masking_it(
    consumer: _Consumer,
) -> None:
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    _in_use, stop = await _source().watch_idle_topics({PROJECTION_TOPIC}, on_first)
    try:
        consumer.getmany_error = RuntimeError("broker went away")
        await _until(lambda: bool(started))
        await _until(lambda: not idle_watched_topics())
        assert started == [PROJECTION_TOPIC]
        assert _attachment(await _health()) == "DEGRADED"
    finally:
        await stop()

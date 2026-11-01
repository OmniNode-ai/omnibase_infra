# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21753: the Kafka path of the consumer-group offset handover, hermetically.

``test_consumer_group_handover_omn21288.py`` drives the handover through an
in-process group-offset substrate that models one partition. The Kafka path
(``_hand_over_on_kafka``) is the code the Redpanda test exercises on a lab
host; these tests run it in CI against a fake broker that holds committed
offsets per group, over several partitions, so that partition enumeration,
negative-offset normalisation, the empty-topic refusal, the per-partition
commit and a second call are all covered without a broker.

Only the consumer factory is replaced. The bus is a real ``EventBusKafka``
(never started), so the group the offsets land on and the physical topic come
from the bus's own resolution, exactly as in production.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal
from unittest.mock import patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    hand_over_committed_offsets,
)
from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)

TOPIC = "onex.evt.omnimarket.live-events.v1"
RETIRED_GROUP = "dev.omnimarket-projections.live-events-writer.consume.v1"
SUCCESSOR_GROUP = "dev.omnimarket.projection_live_events.consume.1.3.0"
FACTORY = (
    "omnibase_infra.handlers.handler_contract_subscribe_preflight."
    "create_offset_consumer"
)

pytestmark = pytest.mark.unit


class _Cluster:
    def __init__(self, partitions: dict[str, set[int]]) -> None:
        self._partitions = partitions

    def partitions_for_topic(self, topic: str) -> set[int] | None:
        return self._partitions.get(topic)


class _Client:
    def __init__(self, cluster: _Cluster) -> None:
        self._cluster = cluster

    async def fetch_all_metadata(self) -> _Cluster:
        return self._cluster


class _FakeBroker:
    """Committed offsets per group, plus each partition's retained range."""

    def __init__(
        self,
        *,
        partitions: dict[str, set[int]],
        first: dict[int, int],
        end: dict[int, int],
    ) -> None:
        self.cluster = _Cluster(partitions)
        self.first = first
        self.end = end
        self.committed: dict[str, dict[TopicPartition, int]] = {}
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.assigned: dict[str, list[TopicPartition]] = {}
        self.commit_calls: list[tuple[str, dict[TopicPartition, int]]] = []


class _FakeConsumer:
    def __init__(self, broker: _FakeBroker, group_id: str) -> None:
        self._broker = broker
        self._group = group_id
        self._client = _Client(broker.cluster)

    async def start(self) -> None:
        self._broker.started.append(self._group)

    async def stop(self) -> None:
        self._broker.stopped.append(self._group)

    def assign(self, partitions: list[TopicPartition]) -> None:
        self._broker.assigned[self._group] = list(partitions)

    async def committed(self, tp: TopicPartition) -> int | None:
        return self._broker.committed.get(self._group, {}).get(tp)

    async def beginning_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {tp: self._broker.first[tp.partition] for tp in partitions}

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {tp: self._broker.end[tp.partition] for tp in partitions}

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self._broker.commit_calls.append((self._group, dict(offsets)))
        self._broker.committed.setdefault(self._group, {}).update(offsets)


def _new_bus() -> EventBusKafka:
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092", environment="dev"
        )
    )


@contextmanager
def _kafka_bus(broker: _FakeBroker) -> Iterator[EventBusKafka]:
    def factory(config: ModelKafkaEventBusConfig, group_id: str) -> _FakeConsumer:
        assert config.bootstrap_servers == bus.bootstrap_servers
        return _FakeConsumer(broker, group_id)

    bus = _new_bus()
    with patch(FACTORY, side_effect=factory):
        yield bus


def _tp(bus: EventBusKafka, partition: int) -> TopicPartition:
    return TopicPartition(bus.resolve_bus_binding(TOPIC).physical_topic, partition)


def _broker(bus_topic: str, partitions: set[int]) -> _FakeBroker:
    return _FakeBroker(
        partitions={bus_topic: partitions},
        first={0: 0, 1: 10, 2: 0, 3: 0},
        end={0: 50, 1: 60, 2: 20, 3: 7},
    )


async def _hand_over(
    bus: EventBusKafka, policy: Literal["earliest", "latest"]
) -> ModelOffsetHandoverResult:
    return await hand_over_committed_offsets(
        bus,
        topic=TOPIC,
        from_group=RETIRED_GROUP,
        to_group=SUCCESSOR_GROUP,
        from_auto_offset_reset=policy,
    )


@pytest.mark.asyncio
async def test_every_partition_of_the_topic_is_handed_over() -> None:
    """Copied, below retention, never committed and negative: one plan each."""
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0, 1, 2, 3})
    with _kafka_bus(broker) as bus:
        retired = {
            _tp(bus, 0): 40,  # inside retention: copied as is
            _tp(bus, 1): 3,  # below the first retained offset (10)
            _tp(bus, 2): -1,  # the broker's "no commit" sentinel
        }  # partition 3: never committed
        broker.committed[RETIRED_GROUP] = retired

        result = await _hand_over(bus, "earliest")

    assert result.committed == {0: 40, 1: 10, 2: 0, 3: 0}
    assert result.kept == ()
    assert broker.assigned[RETIRED_GROUP] == broker.assigned[result.to_group]
    assert [tp.partition for tp in broker.assigned[result.to_group]] == [0, 1, 2, 3]
    ((group, offsets),) = broker.commit_calls
    assert group == result.to_group
    assert {tp.partition: o for tp, o in offsets.items()} == result.committed
    assert group != RETIRED_GROUP, "the retired group is read, never written"


@pytest.mark.asyncio
async def test_latest_policy_starts_an_uncommitted_partition_at_the_end() -> None:
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0, 1, 2})
    with _kafka_bus(broker) as bus:
        broker.committed[RETIRED_GROUP] = {_tp(bus, 0): 12}

        result = await _hand_over(bus, "latest")

    assert result.committed == {0: 12, 1: 60, 2: 20}


@pytest.mark.asyncio
async def test_a_second_handover_keeps_the_successors_own_offsets() -> None:
    """The idempotence rule on the Kafka path: no commit the second time."""
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0, 1})
    with _kafka_bus(broker) as bus:
        broker.committed[RETIRED_GROUP] = {_tp(bus, 0): 5, _tp(bus, 1): 15}
        first = await _hand_over(bus, "earliest")
        # The successor consumes on and moves past the copied position; the
        # retired group's commit is now behind and must not be copied again.
        broker.committed[first.to_group][_tp(bus, 0)] = 30
        second = await _hand_over(bus, "earliest")

    assert first.committed == {0: 5, 1: 15}
    assert second.committed == {}
    assert second.kept == (0, 1)
    assert len(broker.commit_calls) == 1
    assert broker.committed[first.to_group][_tp(bus, 0)] == 30


@pytest.mark.asyncio
async def test_a_topic_with_no_partitions_in_metadata_is_refused() -> None:
    """Nothing to hand over is an error, never a silent join at latest."""
    broker = _FakeBroker(partitions={}, first={}, end={})
    with _kafka_bus(broker) as bus:
        with pytest.raises(ValueError, match="has no partitions in broker metadata"):
            await _hand_over(bus, "earliest")

    assert broker.commit_calls == []
    assert sorted(broker.started) == sorted(broker.stopped), (
        "both short-lived consumers are stopped even when the handover raises"
    )
    assert len(broker.stopped) == 2


@pytest.mark.asyncio
async def test_the_handover_lands_on_the_group_the_bus_will_join() -> None:
    """``to_group`` is the id passed to ``subscribe``; the bus scopes it per topic."""
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0})
    with _kafka_bus(broker) as bus:
        broker.committed[RETIRED_GROUP] = {_tp(bus, 0): 9}
        result = await _hand_over(bus, "earliest")
        expected = bus.resolve_bus_binding(
            TOPIC,
            consumer_group=bus._resolve_effective_group_id(
                SUCCESSOR_GROUP, TOPIC, uuid4(), (TOPIC, SUCCESSOR_GROUP)
            ),
        ).consumer_group

    assert result.to_group == expected
    assert result.to_group != SUCCESSOR_GROUP, "the per-topic scope is applied"
    assert result.from_group == RETIRED_GROUP, "the retired group is used as is"

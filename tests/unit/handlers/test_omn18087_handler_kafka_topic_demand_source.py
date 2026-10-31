# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the group-less reader that splits the boot set and watches idle topics."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers import handler_kafka_topic_demand_source as module
from omnibase_infra.handlers import kafka_idle_topic_watches
from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
    HandlerKafkaTopicDemandSource,
)
from omnibase_infra.handlers.kafka_idle_topic_watches import idle_watched_topics

pytestmark = pytest.mark.unit

BUSY = "onex.cmd.test.busy.v1"
IDLE = "onex.cmd.test.idle.v1"
ABSENT = "onex.cmd.test.not-provisioned.v1"


class FakeConsumer:
    """Holds one partition per topic; ``deliver`` makes a record arrive."""

    instances: list[FakeConsumer] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.offsets: dict[str, tuple[int, int]] = {}
        self.assigned: list[TopicPartition] = []
        self.assignments: list[list[TopicPartition]] = []
        self.seeks: dict[TopicPartition, int] = {}
        self.paused: set[TopicPartition] = set()
        self.arrivals: asyncio.Queue[str] = asyncio.Queue()
        self.getmany_error: Exception | None = None
        self.stopped = False
        FakeConsumer.instances.append(self)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True

    async def topics(self) -> set[str]:
        raise AssertionError("partitions come from the admin client")

    async def beginning_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        # An offset request for a partition never assigned waits on metadata.
        assert set(partitions) <= set(self.assigned)
        return {tp: self.offsets[tp.topic][0] for tp in partitions}

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {tp: self.offsets[tp.topic][1] for tp in partitions}

    def assign(self, partitions: list[TopicPartition]) -> None:
        self.assigned = list(partitions)
        self.assignments.append(list(partitions))

    def seek(self, tp: TopicPartition, offset: int) -> None:
        self.seeks[tp] = offset

    def pause(self, *partitions: TopicPartition) -> None:
        self.paused.update(partitions)

    async def getmany(self, timeout_ms: int = 0) -> dict[TopicPartition, list[object]]:
        if self.getmany_error is not None:
            raise self.getmany_error
        try:
            topic = await asyncio.wait_for(self.arrivals.get(), timeout_ms / 1000)
        except TimeoutError:
            return {}
        return {TopicPartition(topic, 0): [object()]}


class FakeAdmin:
    """Lists and describes the broker's topics, one partition each."""

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


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[int, int]]:
    FakeConsumer.instances.clear()
    offsets: dict[str, tuple[int, int]] = {}

    def build(**kwargs: object) -> FakeConsumer:
        consumer = FakeConsumer(**kwargs)
        consumer.offsets = offsets
        return consumer

    monkeypatch.setattr(kafka_transport, "AIOKafkaConsumer", build)
    monkeypatch.setattr(module, "AIOKafkaAdminClient", lambda **_: FakeAdmin(offsets))
    return offsets


@pytest.fixture(autouse=True)
def _isolated_idle_registry() -> Iterator[None]:
    """OMN-21629: the idle registry is process-wide; no test leaks into the next."""
    kafka_idle_topic_watches._WATCHES.clear()
    yield
    kafka_idle_topic_watches._WATCHES.clear()


def _source() -> HandlerKafkaTopicDemandSource:
    return HandlerKafkaTopicDemandSource(
        ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    )


async def test_split_watches_only_empty_readable_topics(
    broker: dict[str, tuple[int, int]],
) -> None:
    broker.update({BUSY: (0, 5), IDLE: (3, 3)})
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    in_use, stop = await _source().watch_idle_topics({BUSY, IDLE, ABSENT}, on_first)
    consumer = FakeConsumer.instances[0]
    try:
        # A topic that cannot be read stays on the boot path.
        assert in_use == {BUSY, ABSENT}
        assert consumer.assigned == [TopicPartition(IDLE, 0)]
        # Watched from its measured end, so nothing after the measurement is missed.
        assert consumer.seeks == {TopicPartition(IDLE, 0): 3}
        assert consumer.kwargs["group_id"] is None
        assert consumer.kwargs["enable_auto_commit"] is False

        await consumer.arrivals.put(IDLE)
        async with asyncio.timeout(2):
            while not started:
                await asyncio.sleep(0.01)
        assert started == [IDLE]
        assert TopicPartition(IDLE, 0) in consumer.paused
    finally:
        await stop()
    assert consumer.stopped


async def _until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.01)


async def test_a_watched_topic_is_idle_until_its_first_record(
    broker: dict[str, tuple[int, int]],
) -> None:
    """OMN-21629: the health monitor reads which topics still wait for a record."""
    broker.update({BUSY: (0, 5), IDLE: (3, 3)})
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    _in_use, stop = await _source().watch_idle_topics({BUSY, IDLE}, on_first)
    consumer = FakeConsumer.instances[0]
    try:
        assert idle_watched_topics() == {IDLE}
        await consumer.arrivals.put(IDLE)
        await _until(lambda: not idle_watched_topics())
        assert started == [IDLE]
    finally:
        await stop()


async def test_stop_before_the_first_record_clears_the_idle_topics(
    broker: dict[str, tuple[int, int]],
) -> None:
    """A stopped watch masks nothing: its contracts will never start on demand."""
    broker.update({IDLE: (3, 3)})

    async def on_first(topic: str) -> None:
        raise AssertionError("no record arrived")

    _in_use, stop = await _source().watch_idle_topics({IDLE}, on_first)
    assert idle_watched_topics() == {IDLE}
    await stop()
    assert idle_watched_topics() == frozenset()
    # Stopping twice is harmless.
    await stop()
    assert idle_watched_topics() == frozenset()


async def test_a_cancelled_watch_task_clears_the_idle_topics(
    broker: dict[str, tuple[int, int]],
) -> None:
    """A watch that dies without stop() being called stops masking too."""
    broker.update({IDLE: (3, 3)})

    async def on_first(topic: str) -> None:
        raise AssertionError("no record arrived")

    _in_use, stop = await _source().watch_idle_topics({IDLE}, on_first)
    try:
        (task,) = [
            t
            for t in asyncio.all_tasks()
            if t.get_name() == "boot-subscription-demand-watch"
        ]
        task.cancel()
        await _until(lambda: not idle_watched_topics())
    finally:
        await stop()


async def test_a_failed_watch_starts_every_topic_and_clears_the_idle_set(
    broker: dict[str, tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback path reports contract names and leaves nothing idle.

    Under a lane namespace the consumer reads PHYSICAL topic names; the
    fallback that starts every deferred contract must map them back, or the
    start would name a topic no contract subscribes to and the idle set would
    keep a name that is never discarded.
    """
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr9")
    broker.update({f"prepr9.{IDLE}": (0, 0)})
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    _in_use, stop = await _source().watch_idle_topics({IDLE}, on_first)
    consumer = FakeConsumer.instances[0]
    try:
        consumer.getmany_error = RuntimeError("broker went away")
        await _until(lambda: bool(started))
        assert started == [IDLE]
        assert idle_watched_topics() == frozenset()
    finally:
        await stop()


async def test_a_demand_start_that_fails_still_leaves_the_idle_set(
    broker: dict[str, tuple[int, int]],
) -> None:
    """A failed start is judged by the live registry afterwards, so it is reported."""
    broker.update({IDLE: (3, 3)})
    attempted: list[str] = []

    async def on_first(topic: str) -> None:
        attempted.append(topic)
        raise RuntimeError("contract failed to start")

    _in_use, stop = await _source().watch_idle_topics({IDLE}, on_first)
    consumer = FakeConsumer.instances[0]
    try:
        await consumer.arrivals.put(IDLE)
        await _until(lambda: not idle_watched_topics())
        assert attempted == [IDLE]
    finally:
        await stop()


async def test_two_sources_keep_their_own_idle_topics(
    broker: dict[str, tuple[int, int]],
) -> None:
    """Stopping one watch never clears a topic another watch still waits on."""
    other = "onex.cmd.test.other-idle.v1"
    broker.update({IDLE: (3, 3), other: (1, 1)})

    async def on_first(topic: str) -> None:
        return None

    _, stop_first = await _source().watch_idle_topics({IDLE}, on_first)
    _, stop_second = await _source().watch_idle_topics({other}, on_first)
    try:
        assert idle_watched_topics() == {IDLE, other}
        await stop_first()
        assert idle_watched_topics() == {other}
    finally:
        await stop_second()
    assert idle_watched_topics() == frozenset()


async def test_a_busy_topic_is_never_idle(
    broker: dict[str, tuple[int, int]],
) -> None:
    broker.update({BUSY: (0, 5)})

    async def on_first(topic: str) -> None:
        raise AssertionError("nothing is watched")

    await _source().watch_idle_topics({BUSY, ABSENT}, on_first)
    assert idle_watched_topics() == frozenset()


async def test_namespaced_lane_reads_physical_topics_and_reports_contract_names(
    broker: dict[str, tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr9")
    broker.update({f"prepr9.{BUSY}": (0, 1), f"prepr9.{IDLE}": (0, 0), IDLE: (0, 9)})
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    in_use, stop = await _source().watch_idle_topics({BUSY, IDLE}, on_first)
    consumer = FakeConsumer.instances[0]
    try:
        # The un-namespaced IDLE topic holds records, but it is not this lane's.
        assert in_use == {BUSY}
        assert consumer.assigned == [TopicPartition(f"prepr9.{IDLE}", 0)]
        await consumer.arrivals.put(f"prepr9.{IDLE}")
        async with asyncio.timeout(2):
            while not started:
                await asyncio.sleep(0.01)
        assert started == [IDLE]
    finally:
        await stop()


async def test_nothing_idle_starts_no_watch(
    broker: dict[str, tuple[int, int]],
) -> None:
    broker.update({BUSY: (0, 5)})

    async def on_first(topic: str) -> None:
        raise AssertionError("no idle topic to start")

    in_use, stop = await _source().watch_idle_topics({BUSY}, on_first)
    assert in_use == {BUSY}
    assert FakeConsumer.instances[0].stopped
    await stop()


async def test_unmeasurable_broker_keeps_every_topic_on_the_boot_path(
    broker: dict[str, tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**_: object) -> FakeAdmin:
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(module, "AIOKafkaAdminClient", refuse)

    async def on_first(topic: str) -> None:
        raise AssertionError("nothing is watched")

    in_use, stop = await _source().watch_idle_topics({BUSY, IDLE}, on_first)
    assert in_use == {BUSY, IDLE}
    await stop()


async def test_dead_watch_starts_every_deferred_topic(
    broker: dict[str, tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker.update({IDLE: (3, 3)})
    started: list[str] = []

    async def on_first(topic: str) -> None:
        started.append(topic)

    _, stop = await _source().watch_idle_topics({IDLE}, on_first)
    consumer = FakeConsumer.instances[0]

    async def broken(timeout_ms: int = 0) -> dict[TopicPartition, list[object]]:
        raise RuntimeError("broker disconnected")

    monkeypatch.setattr(consumer, "getmany", broken)
    try:
        async with asyncio.timeout(2):
            while not started:
                await asyncio.sleep(0.01)
        assert started == [IDLE]
    finally:
        await stop()


async def test_empty_boot_set_starts_no_reader(
    broker: dict[str, tuple[int, int]],
) -> None:
    # A runtime profile that owns no auto-wired contract (the laptop profile's
    # tenant-projection-writer, whose contracts live in omnimarket) hands over
    # an empty boot set. Starting a group-less reader and stopping it with no
    # partition assigned raises CancelledError out of aiokafka's coordinator
    # close, which ended the kernel in a restart loop (OMN-18087, run
    # 40881579955). No reader may be started when there is nothing to read.
    async def on_first(topic: str) -> None:
        raise AssertionError(f"nothing is watched, got {topic}")

    in_use, stop = await _source().watch_idle_topics(set(), on_first)

    assert in_use == frozenset()
    assert FakeConsumer.instances == []
    await stop()


async def test_boot_set_of_unprovisioned_topics_starts_no_reader(
    broker: dict[str, tuple[int, int]],
) -> None:
    broker.update({BUSY: (0, 5)})

    async def on_first(topic: str) -> None:
        raise AssertionError(f"nothing is watched, got {topic}")

    in_use, stop = await _source().watch_idle_topics({ABSENT}, on_first)

    # Unreadable topics stay on the boot path, as before.
    assert in_use == {ABSENT}
    assert FakeConsumer.instances == []
    await stop()

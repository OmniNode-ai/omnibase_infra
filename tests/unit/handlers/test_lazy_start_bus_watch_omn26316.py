# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A message a client writes straight to the broker wakes a dormant consumer (OMN-26316).

Measured on the dev runtime (boot of 2026-11-23T19:09:12Z):
``node_house_routing_overlay_effect`` was discovered at 19:10:06Z and no log line
afterwards names it or its command topic
``onex.cmd.omnimarket.house-routing-overlay-write.v1``. The lane left it dormant, so
nothing subscribes to the topic; the client writes the command to the broker, which is
neither the ingress topic nor the runtime's own publish path, so no front-door wake
ever ran.

The watch is one group-less reader of the dormant contracts' topics; the ingress wakes
the consumers of a topic on its first message. The fake broker below records what the
reader asks of it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.handlers import handler_lazy_start_ingress as ingress_module
from omnibase_infra.handlers import lazy_start_dormant_topic_watch as watch_module
from omnibase_infra.handlers.handler_lazy_start_ingress import HandlerLazyStartIngress
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    StoreLazyStartWaker,
)
from omnibase_infra.handlers.lazy_start_dormant_topic_watch import (
    LazyStartDormantTopicWatch,
)
from omnibase_infra.models.lazy_start import ModelLazyStartPolicy
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)

pytestmark = pytest.mark.unit

HOUSE = "node_house_routing_overlay_effect"
HOUSE_COMMAND = "onex.cmd.test.house-routing-overlay-write.v1"
HOUSE_WRITTEN = "onex.evt.test.house-routing-overlay-written.v1"
OTHER = "node_other_dormant"
OTHER_COMMAND = "onex.cmd.test.other-dormant.v1"
OTHER_WRITTEN = "onex.evt.test.other-dormant-done.v1"
ABSENT_COMMAND = "onex.cmd.test.not-on-broker.v1"
BOOT_COMMAND = "onex.cmd.test.boot-set.v1"
BOOT_WRITTEN = "onex.evt.test.boot-set-done.v1"
CALLS: list[tuple[str, str]] = []


class FakeBroker:
    """What the watch reads: topic names, partition counts and end offsets."""

    def __init__(self, partitions_of: dict[str, int], end: int = 7) -> None:
        self.partitions_of = partitions_of
        self.end = end
        self.describe_sizes: list[int] = []
        self.consumers: list[FakeConsumer] = []
        self.records: asyncio.Queue[dict[Any, list[object]]] = asyncio.Queue()


class FakeAdmin:
    broker: FakeBroker

    def __init__(self, **_kwargs: object) -> None:
        pass

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def list_topics(self) -> list[str]:
        return list(self.broker.partitions_of)

    async def describe_topics(self, topics: list[str]) -> list[dict[str, object]]:
        self.broker.describe_sizes.append(len(topics))
        return [
            {
                "topic": t,
                "error_code": 0,
                "partitions": [
                    {"partition": p} for p in range(self.broker.partitions_of[t])
                ],
            }
            for t in topics
        ]


class FakeConsumer:
    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker
        self.assigned: list[Any] = []
        self.seeks: dict[Any, int] = {}
        self.paused: list[Any] = []
        self.stopped = False
        self.group_id: str | None = "unset"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True

    def assign(self, partitions: list[Any]) -> None:
        self.assigned = list(partitions)

    async def end_offsets(self, partitions: list[Any]) -> dict[Any, int]:
        return dict.fromkeys(partitions, self.broker.end)

    def seek(self, tp: Any, offset: int) -> None:
        self.seeks[tp] = offset

    def pause(self, *partitions: Any) -> None:
        self.paused.extend(partitions)

    async def getmany(self, timeout_ms: int = 0) -> dict[Any, list[object]]:
        try:
            return await asyncio.wait_for(self.broker.records.get(), timeout=0.05)
        except TimeoutError:
            return {}


class FakeKafkaBus(EventBusKafka):
    """Kafka-typed stand-in; never opens a transport."""

    def __init__(self) -> None:
        self._environment = "local"

    @property
    def config(self) -> Any:
        return SimpleNamespace(
            bootstrap_servers="broker:9092", consumer_poll_timeout_ms=10
        )

    async def publish(self, *args: object, **kwargs: object) -> None:
        return None


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> FakeBroker:
    fake = FakeBroker({HOUSE_COMMAND: 2, OTHER_COMMAND: 1})
    admin = type("BoundAdmin", (FakeAdmin,), {"broker": fake})
    monkeypatch.setattr(watch_module, "AIOKafkaAdminClient", admin)
    monkeypatch.setattr(watch_module, "build_aiokafka_auth_kwargs", lambda _c: {})

    def make_consumer(_config: object, group_id: str | None = None) -> FakeConsumer:
        consumer = FakeConsumer(fake)
        consumer.group_id = group_id
        fake.consumers.append(consumer)
        return consumer

    monkeypatch.setattr(watch_module, "create_offset_consumer", make_consumer)
    return fake


async def _container() -> ModelONEXContainer:
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, FakeKafkaBus())
    return container


def _record_for(topic: str, partition: int = 0) -> dict[Any, list[object]]:
    from aiokafka.structs import TopicPartition

    return {TopicPartition(topic, partition): [object()]}


async def test_the_watch_reads_from_the_end_without_a_group_and_reports_a_first_record(
    broker: FakeBroker,
) -> None:
    watch = LazyStartDormantTopicWatch(await _container())
    arrived: list[str] = []

    async def on_first_record(topic: str) -> None:
        arrived.append(topic)

    stop = await watch.watch({HOUSE_COMMAND, OTHER_COMMAND}, on_first_record, 100)
    try:
        (consumer,) = broker.consumers
        # Group-less and commit-free: it moves no offset a real consumer reads from.
        assert consumer.group_id is None
        # Every partition, from the current end: a record that was already retained is
        # not a first request.
        assert len(consumer.assigned) == 3
        assert set(consumer.seeks.values()) == {broker.end}
        await broker.records.put(_record_for(HOUSE_COMMAND, 1))
        for _ in range(100):
            if arrived:
                break
            await asyncio.sleep(0.02)
        assert arrived == [HOUSE_COMMAND]
        # The other topic stays watched and unreported.
        await asyncio.sleep(0.2)
        assert arrived == [HOUSE_COMMAND]
    finally:
        await stop()
    assert consumer.stopped


async def test_a_topic_the_broker_does_not_hold_is_not_watched(
    broker: FakeBroker,
) -> None:
    watch = LazyStartDormantTopicWatch(await _container())

    async def on_first_record(topic: str) -> None:
        raise AssertionError(topic)

    stop = await watch.watch({HOUSE_COMMAND, ABSENT_COMMAND}, on_first_record, 100)
    try:
        (consumer,) = broker.consumers
        assert {tp.topic for tp in consumer.assigned} == {HOUSE_COMMAND}
    finally:
        await stop()


async def test_a_dormant_set_of_thousands_is_described_in_bounded_requests(
    broker: FakeBroker,
) -> None:
    topics = {f"onex.cmd.test.t{i}.v1" for i in range(250)}
    broker.partitions_of = dict.fromkeys(topics, 1)
    watch = LazyStartDormantTopicWatch(await _container())

    async def on_first_record(topic: str) -> None:
        return None

    stop = await watch.watch(topics, on_first_record, 100)
    try:
        assert broker.describe_sizes == [100, 100, 50]
    finally:
        await stop()


async def test_with_no_topic_on_the_broker_there_is_nothing_to_read(
    broker: FakeBroker,
) -> None:
    broker.partitions_of = {}
    watch = LazyStartDormantTopicWatch(await _container())

    async def on_first_record(topic: str) -> None:
        return None

    stop = await watch.watch({HOUSE_COMMAND}, on_first_record, 100)
    await stop()
    assert broker.consumers == []


async def test_a_bus_that_is_not_kafka_has_no_broker_to_watch() -> None:
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, object())
    with pytest.raises(TypeError):
        LazyStartDormantTopicWatch(container)


class RecordingTransaction:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def build(self, name: str) -> None:
        CALLS.append(("build", name))

    async def attach(self, name: str) -> None:
        CALLS.append(("attach", name))

    async def rollback(self, name: str) -> None:
        CALLS.append(("rollback", name))

    async def deliver(self, name: str, topic: str, envelope: object) -> object:
        raise AssertionError("a broker message is never delivered as an envelope")


class RecordingWatch:
    """The dormant-topic watch, capturing what the ingress arms and calling back on cue."""

    instances: list[RecordingWatch] = []

    def __init__(self, _container: object) -> None:
        self.topics: frozenset[str] = frozenset()
        self.on_first_record: Callable[[str], Awaitable[None]] | None = None
        self.stopped = False
        RecordingWatch.instances.append(self)

    async def watch(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
        request_chunk: int,
    ) -> Callable[[], Awaitable[None]]:
        self.chunk = request_chunk
        self.topics = frozenset(topics)
        self.on_first_record = on_first_record

        async def stop() -> None:
            self.stopped = True

        return stop


def _contract(
    tmp_path: Path, name: str, subscribe: str, publish: str
) -> ModelDiscoveredContract:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"name: {name}\n"
        "node_type: EFFECT_GENERIC\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "event_bus:\n"
        f'  subscribe_topics: ["{subscribe}"]\n'
        f'  publish_topics: ["{publish}"]\n',
        encoding="utf-8",
    )
    return _parse_contract(
        contract_path=path,
        entry_point_name=name,
        package_name="fixture",
        package_version="0.0.0",
    )


@pytest.fixture
def _ingress_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    CALLS.clear()
    RecordingWatch.instances.clear()
    monkeypatch.setattr(
        ingress_module, "LazyStartRuntimeTransaction", RecordingTransaction
    )
    monkeypatch.setattr(
        ingress_module, "LazyStartDormantTopicWatch", RecordingWatch, raising=False
    )
    yield
    StoreLazyStartDormant.clear()


@pytest.mark.usefixtures("_ingress_env")
async def test_the_ingress_arms_the_dormant_topics_and_a_first_message_wakes_the_consumer(
    tmp_path: Path,
) -> None:
    boot = _contract(tmp_path, "node_boot", BOOT_COMMAND, BOOT_WRITTEN)
    house = _contract(tmp_path, HOUSE, HOUSE_COMMAND, HOUSE_WRITTEN)
    other = _contract(tmp_path, OTHER, OTHER_COMMAND, OTHER_WRITTEN)
    StoreLazyStartDormant.record(
        ModelLazyStartPolicy(
            ingress_contract="node_lazy_start_ingress", boot_contracts=("node_boot",)
        ),
        (house, other),
        None,
        (boot,),
    )
    handler = HandlerLazyStartIngress(await _container())
    assert handler._watch_task is not None
    await handler._watch_task
    (watch,) = RecordingWatch.instances
    assert watch.chunk == 100
    # Only dormant consumers are watched; a boot contract's topic has a subscription.
    assert watch.topics == {HOUSE_COMMAND, OTHER_COMMAND}
    assert CALLS == []
    assert watch.on_first_record is not None
    await watch.on_first_record(HOUSE_COMMAND)
    assert CALLS == [("build", HOUSE), ("attach", HOUSE)]
    # A second message to a topic whose consumer serves wakes nothing again.
    await watch.on_first_record(HOUSE_COMMAND)
    assert CALLS == [("build", HOUSE), ("attach", HOUSE)]
    await handler.shutdown()
    assert watch.stopped
    assert StoreLazyStartWaker.waker() is None


@pytest.mark.usefixtures("_ingress_env")
async def test_a_wake_that_fails_for_a_message_fails_that_contract_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    house = _contract(tmp_path, HOUSE, HOUSE_COMMAND, HOUSE_WRITTEN)

    class Failing(RecordingTransaction):
        async def attach(self, name: str) -> None:
            raise ConnectionError("join refused")

    monkeypatch.setattr(ingress_module, "LazyStartRuntimeTransaction", Failing)
    StoreLazyStartDormant.record(
        ModelLazyStartPolicy(ingress_contract="node_lazy_start_ingress", max_retries=0),
        (house,),
        None,
        (),
    )
    handler = HandlerLazyStartIngress(await _container())
    assert handler._watch_task is not None
    await handler._watch_task
    (watch,) = RecordingWatch.instances
    assert watch.on_first_record is not None
    await watch.on_first_record(HOUSE_COMMAND)
    front_door = cast("Any", handler._front_door)
    assert set(front_door.wake_failures) == {HOUSE}

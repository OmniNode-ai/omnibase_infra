# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21229 (H3b): a subscription has a lifecycle the runtime can trust.

Four properties, each failing before the change:

1. a subscribe that fails to start leaves no callback, and the retry
   dispatches each record once;
2. the returned handle pauses, drains and stops in that order, and stop after
   drain lets the in-flight callback finish instead of cancelling it;
3. the serial path commits only completed records, and a shutdown with a
   fetched batch in flight commits exactly the completed ones;
4. the in-memory bus has the same handle.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from omnibase_infra.enums.enum_subscription_phase import EnumSubscriptionPhase
from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.event_bus.model_subscription_handle import (
    ModelSubscriptionHandle,
)

pytestmark = pytest.mark.unit

TOPIC = "omn21229-lifecycle"
GROUP = "test.omnimarket.omn21229.consume.v1"
PARTITION = 0
BASE = 100


class _FakeConsumer:
    """Serves one batch, records commits, seeks and pauses."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.commits: list[dict[TopicPartition, int]] = []
        self.seeks: list[tuple[TopicPartition, int]] = []
        self.paused = False
        self.stopped = False

    async def getmany(self, *_: Any, **__: Any) -> dict[TopicPartition, list[Any]]:
        if not self._messages or self.paused:
            await asyncio.sleep(0.01)
            return {}
        batch, self._messages = self._messages, []
        return {TopicPartition(TOPIC, PARTITION): batch}

    def assignment(self) -> set[TopicPartition]:
        return {TopicPartition(TOPIC, PARTITION)}

    def pause(self, *_: TopicPartition) -> None:
        self.paused = True

    def seek(self, topic_partition: TopicPartition, offset: int) -> None:
        self.seeks.append((topic_partition, offset))

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self.commits.append(dict(offsets))

    async def stop(self) -> None:
        self.stopped = True

    def committed(self) -> list[int]:
        return [int(v) for c in self.commits for v in c.values()]

    def feed(self, *messages: Any) -> None:
        """Queue records for the next ``getmany``."""
        self._messages.extend(messages)


def _msg(offset: int) -> MagicMock:
    msg = MagicMock()
    msg.topic = TOPIC
    msg.partition = PARTITION
    msg.offset = offset
    msg.key = f"omn21229-{offset}".encode()
    msg.value = b'{"payload": "ok"}'
    msg.headers = ()
    return msg


def _config(**overrides: Any) -> ModelKafkaEventBusConfig:
    base: dict[str, Any] = {
        "bootstrap_servers": "localhost:9092",
        "environment": "dev",
        "dead_letter_topic": "dlq-events",
    }
    base.update(overrides)
    return ModelKafkaEventBusConfig(**base)


def _producer() -> AsyncMock:
    producer = AsyncMock()
    producer._closed = False

    async def _send(topic: str, **_: Any) -> asyncio.Future[Any]:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        future.set_result(SimpleNamespace(topic=topic, partition=0, offset=1))
        return future

    producer.send = AsyncMock(side_effect=_send)
    return producer


def _consumer_starter(
    bus: EventBusKafka, consumer: _FakeConsumer
) -> Callable[..., Awaitable[None]]:
    """Stand in for the broker join: install the fake consumer and its loop."""

    async def _start(topic: str, group_id: str, **_: Any) -> None:
        key = (topic, group_id)
        bus._pending_consumer_keys.discard(key)
        bus._group_consumers[key] = cast("AIOKafkaConsumer", consumer)
        bus._group_consumer_tasks[key] = asyncio.create_task(
            bus._consume_loop(topic, group_id, uuid4())
        )
        bus._refresh_topic_consumer_views(topic)

    return _start


# --------------------------------------------------------------------------
# 1. A failed start leaves no callback; the retry dispatches once.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_start_leaves_no_callback_and_retry_dispatches_once() -> None:
    dispatched: list[int] = []
    consumer = _FakeConsumer([_msg(BASE)])

    async def _callback(_message: ModelEventMessage) -> None:
        dispatched.append(1)

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()
        starter = _consumer_starter(bus, consumer)

        async def _refuse(topic: str, group_id: str, **_: Any) -> None:
            # The real start drops its reservation before it raises.
            bus._pending_consumer_keys.discard((topic, group_id))
            raise InfraConnectionError("broker refused")

        failing = AsyncMock(side_effect=_refuse)
        with patch.object(bus, "_start_consumer_for_topic_unlocked", failing):
            with pytest.raises(InfraConnectionError):
                await bus.subscribe(TOPIC, group_id=GROUP, on_message=_callback)
        assert bus._subscribers.get(TOPIC, []) == [], (
            "a start failure left its callback registered"
        )

        with patch.object(bus, "_start_consumer_for_topic_unlocked", starter):
            handle = await bus.subscribe(TOPIC, group_id=GROUP, on_message=_callback)
        await asyncio.sleep(0.3)
        await handle.stop()
        await bus.close()

    assert dispatched == [1], f"the record was dispatched {len(dispatched)} times"


@pytest.mark.asyncio
async def test_cancelled_start_leaves_no_callback() -> None:
    async def _callback(_message: ModelEventMessage) -> None:
        return None

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()

        async def _hang(*_: Any, **__: Any) -> None:
            await asyncio.sleep(60)

        with patch.object(bus, "_start_consumer_for_topic_unlocked", _hang):
            task = asyncio.create_task(
                bus.subscribe(TOPIC, group_id=GROUP, on_message=_callback)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert bus._subscribers.get(TOPIC, []) == []
        await bus.close()


# --------------------------------------------------------------------------
# 2. The handle: pause, drain, stop, in that order; stop does not cut a callback.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_runs_pause_drain_stop_in_order_whatever_is_called() -> None:
    order: list[str] = []

    def _step(name: str) -> Callable[[], Awaitable[None]]:
        async def _run() -> None:
            order.append(name)

        return _run

    def _handle() -> ModelSubscriptionHandle:
        return ModelSubscriptionHandle.bind(
            topic="t",
            consumer_group="g",
            subscription_key="s",
            pause=_step("pause"),
            drain=_step("drain"),
            stop=_step("stop"),
        )

    handle = _handle()
    await handle.drain()
    assert order == ["pause", "drain"], "drain must pause first"
    await handle.stop()
    await handle.stop()
    assert order == ["pause", "drain", "stop"], "each step runs once, in order"
    assert handle.phase is EnumSubscriptionPhase.STOPPED

    order.clear()
    await _handle()()
    assert order == ["pause", "drain", "stop"], "awaiting the handle is stop"


@pytest.mark.asyncio
async def test_stop_removes_the_callback_even_when_drain_raises() -> None:
    stopped: list[bool] = []

    async def _ok() -> None:
        return None

    async def _bad_drain() -> None:
        raise RuntimeError("drain failed")

    async def _stop() -> None:
        stopped.append(True)

    handle = ModelSubscriptionHandle.bind(
        topic="t",
        consumer_group="g",
        subscription_key="s",
        pause=_ok,
        drain=_bad_drain,
        stop=_stop,
    )
    with pytest.raises(RuntimeError):
        await handle.stop()
    assert stopped == [True]


@pytest.mark.asyncio
async def test_stop_lets_the_in_flight_callback_finish() -> None:
    started = asyncio.Event()
    outcome: list[str] = []
    consumer = _FakeConsumer([_msg(BASE)])

    async def _callback(_message: ModelEventMessage) -> None:
        started.set()
        try:
            await asyncio.sleep(0.3)
            outcome.append("completed")
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()
        with patch.object(
            bus, "_start_consumer_for_topic_unlocked", _consumer_starter(bus, consumer)
        ):
            handle = await bus.subscribe(TOPIC, group_id=GROUP, on_message=_callback)
        await asyncio.wait_for(started.wait(), timeout=2)
        await handle.pause()
        assert consumer.paused, "pause must stop the consumer fetching"
        await handle.drain()
        assert outcome == ["completed"], "drain returned before the callback did"
        await handle.stop()
        await bus.close()

    assert outcome == ["completed"], f"the in-flight callback was {outcome}"
    assert consumer.committed() == [BASE + 1]


# --------------------------------------------------------------------------
# 3. Completion-based commits on the serial path.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serial_path_commits_each_completed_record_not_the_fetch_position() -> (
    None
):
    consumer = _FakeConsumer([_msg(BASE + i) for i in range(3)])
    done: list[int] = []

    async def _callback(message: ModelEventMessage) -> None:
        done.append(int(message.offset))

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()
        with patch.object(
            bus, "_start_consumer_for_topic_unlocked", _consumer_starter(bus, consumer)
        ):
            handle = await bus.subscribe(TOPIC, group_id=GROUP, on_message=_callback)
        await asyncio.sleep(0.4)
        await handle.stop()
        await bus.close()

    assert done == [BASE, BASE + 1, BASE + 2]
    assert consumer.committed() == [BASE + 1, BASE + 2, BASE + 3]


def test_no_consumer_is_built_with_auto_commit() -> None:
    bus = EventBusKafka(config=_config(enable_auto_commit=True))
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer"
    ) as consumer_cls:
        bus._build_consumer(
            TOPIC, f"{GROUP}.__t.{TOPIC}", "inst", "latest", group_id=GROUP
        )
    assert consumer_cls.call_args.kwargs["enable_auto_commit"] is False


@pytest.mark.asyncio
async def test_shutdown_with_a_fetched_batch_in_flight_commits_only_completed() -> None:
    """The SIGTERM case: close() lands while the first of three records runs."""
    started = asyncio.Event()
    outcome: list[tuple[int, str]] = []
    consumer = _FakeConsumer([_msg(BASE + i) for i in range(3)])

    async def _callback(message: ModelEventMessage) -> None:
        offset = int(message.offset)
        started.set()
        try:
            await asyncio.sleep(0.4)
            outcome.append((offset, "completed"))
        except asyncio.CancelledError:
            outcome.append((offset, "cancelled"))
            raise

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config(consumer_shutdown_drain_seconds=5.0))
        await bus.start()
        with patch.object(
            bus, "_start_consumer_for_topic_unlocked", _consumer_starter(bus, consumer)
        ):
            await bus.subscribe(TOPIC, group_id=GROUP, on_message=_callback)
        await asyncio.wait_for(started.wait(), timeout=2)
        await bus.close()

    assert outcome == [(BASE, "completed")], (
        f"the in-flight callback must finish and the rest stay unstarted: {outcome}"
    )
    assert consumer.committed() == [BASE + 1], (
        "the committed offset must cover the completed record and nothing else: "
        f"{consumer.committed()}"
    )
    assert (TopicPartition(TOPIC, PARTITION), BASE + 1) in consumer.seeks


@pytest.mark.asyncio
async def test_paused_handle_gets_no_record_while_its_group_sibling_runs() -> None:
    """Pause is per subscription: a paused-and-drained handle is handed nothing.

    Two subscriptions share one topic and one consumer group, so one consumer
    serves both and dispatch fans every record out to both. Pausing one of them
    must keep the next record away from it while the other still receives it.
    """
    consumer = _FakeConsumer([])
    paused_got: list[int] = []
    sibling_got: list[int] = []

    async def _paused_callback(message: ModelEventMessage) -> None:
        paused_got.append(int(message.offset))

    async def _sibling_callback(message: ModelEventMessage) -> None:
        sibling_got.append(int(message.offset))

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()
        with patch.object(
            bus, "_start_consumer_for_topic_unlocked", _consumer_starter(bus, consumer)
        ):
            paused = await bus.subscribe(
                TOPIC, group_id=GROUP, on_message=_paused_callback
            )
            sibling = await bus.subscribe(
                TOPIC, group_id=GROUP, on_message=_sibling_callback
            )
        consumer.feed(_msg(BASE))
        await asyncio.sleep(0.2)
        assert paused_got == [BASE]
        assert sibling_got == [BASE]

        await paused.pause()
        await paused.drain()
        assert not consumer.paused, "the sibling is live: the consumer keeps fetching"

        consumer.feed(_msg(BASE + 1))
        await asyncio.sleep(0.2)
        await sibling.stop()
        await paused.stop()
        await bus.close()

    assert sibling_got == [BASE, BASE + 1], (
        f"the live sibling lost a record: {sibling_got}"
    )
    assert paused_got == [BASE], f"the paused handle received a record: {paused_got}"


@pytest.mark.asyncio
async def test_inmemory_paused_handle_gets_no_record_while_its_sibling_runs() -> None:
    bus = EventBusInmemory()
    await bus.start()
    paused_got: list[bytes] = []
    sibling_got: list[bytes] = []

    async def _paused_callback(message: ModelEventMessage) -> None:
        paused_got.append(message.value)

    async def _sibling_callback(message: ModelEventMessage) -> None:
        sibling_got.append(message.value)

    paused = await bus.subscribe(TOPIC, group_id=GROUP, on_message=_paused_callback)
    sibling = await bus.subscribe(TOPIC, group_id=GROUP, on_message=_sibling_callback)
    await bus.publish(TOPIC, None, b"one")
    await paused.pause()
    await paused.drain()
    await bus.publish(TOPIC, None, b"two")
    await sibling.stop()
    await paused.stop()
    await bus.close()

    assert sibling_got == [b"one", b"two"]
    assert paused_got == [b"one"]


# --------------------------------------------------------------------------
# 4. The in-memory bus has the same handle.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_handle_pauses_drains_and_stops() -> None:
    bus = EventBusInmemory()
    await bus.start()
    started = asyncio.Event()
    outcome: list[str] = []

    async def _callback(_message: ModelEventMessage) -> None:
        started.set()
        try:
            await asyncio.sleep(0.2)
            outcome.append("completed")
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise

    handle = await bus.subscribe(TOPIC, group_id=GROUP, on_message=_callback)
    publisher = asyncio.create_task(bus.publish(TOPIC, None, b"one"))
    await asyncio.wait_for(started.wait(), timeout=2)
    await handle.pause()
    await bus.publish(TOPIC, None, b"two")
    await handle.stop()
    await publisher
    await bus.publish(TOPIC, None, b"three")
    await bus.close()

    assert outcome == ["completed"], (
        f"the in-flight callback finished; nothing was delivered after pause: {outcome}"
    )
    assert handle.phase is EnumSubscriptionPhase.STOPPED

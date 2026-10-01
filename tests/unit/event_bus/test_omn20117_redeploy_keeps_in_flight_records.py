# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20117: a redeploy must not lose a record whose handler is still running.

Measured on the ``.201`` dev lane, 2026-09-30T11:15Z, by a warm restart of
``omninode-runtime-effects`` while an ``onex delegate`` was in flight:

* the delegate-skill group's committed offset went from 4847 to 4853 across
  the restart with six commands still running, because the concurrent
  (OMN-18852) path runs under auto-commit, which commits the FETCH position,
  and nothing redelivered them;
* the six handlers that did finish during shutdown could not publish their
  terminals: ``close()`` marks the bus stopped before it lets the consume tasks
  drain, so every publish raised ``Event bus not started`` and the commands
  went to the DLQ instead of answering their callers.

Four properties are pinned here:

1. a subscription that declares concurrency is built without auto-commit and
   commits a low watermark that is never past an unfinished record;
2. ``close()`` lets running dispatches finish and publish before it refuses
   publishes;
3. a dispatch still running when the drain ends is cancelled, may answer the
   cancellation with a terminal (which is published and committed), and is
   otherwise never committed past, so it is redelivered;
4. the serial path pins the fetch position at the first unprocessed record of
   a batch when shutdown cuts the batch short.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.concurrent_commit_ledger import ConcurrentCommitLedger
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

pytestmark = pytest.mark.unit

TOPIC = "omn20117-delegate-skill"
GROUP = "test.omnimarket.omn20117.consume.v1"
PARTITION = 0
BASE = 4847


class _FakeConsumer:
    """Serves records from a queue and records commits and seeks."""

    def __init__(self, messages: list[Any], *, batch_size: int) -> None:
        self._messages = list(messages)
        self._batch_size = batch_size
        self.commits: list[dict[TopicPartition, int]] = []
        self.seeks: list[tuple[TopicPartition, int]] = []
        self.stopped = False

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[Any]]:
        if not self._messages:
            await asyncio.sleep(0.01)
            return {}
        batch = self._messages[: self._batch_size]
        del self._messages[: self._batch_size]
        return {TopicPartition(TOPIC, PARTITION): batch}

    def assignment(self) -> set[TopicPartition]:
        return {TopicPartition(TOPIC, PARTITION)}

    def seek(self, topic_partition: TopicPartition, offset: int) -> None:
        self.seeks.append((topic_partition, offset))

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self.commits.append(dict(offsets))

    async def stop(self) -> None:
        self.stopped = True

    def committed_offsets(self) -> list[int]:
        return [
            int(getattr(value, "offset", value))
            for commit in self.commits
            for value in commit.values()
        ]


def _msg(offset: int) -> MagicMock:
    msg = MagicMock()
    msg.topic = TOPIC
    msg.partition = PARTITION
    msg.offset = offset
    msg.key = f"omn20117-{offset}".encode()
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
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer._closed = False

    async def _send(topic: str, **_: Any) -> asyncio.Future[Any]:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        future.set_result(SimpleNamespace(topic=topic, partition=0, offset=1))
        return future

    producer.send = AsyncMock(side_effect=_send)
    producer.send_and_wait = AsyncMock()
    return producer


# --------------------------------------------------------------------------
# 1. The ledger itself.
# --------------------------------------------------------------------------


def test_ledger_never_passes_an_unfinished_offset() -> None:
    ledger = ConcurrentCommitLedger()
    for offset in (10, 11, 12, 13):
        ledger.dispatched(PARTITION, offset)
    for offset in (11, 12, 13):
        ledger.finished(PARTITION, offset)
    assert ledger.advanced() == {PARTITION: 10}, "10 is still running"
    ledger.committed(PARTITION, 10)
    assert ledger.advanced() == {}, "nothing moved since the last commit"
    ledger.finished(PARTITION, 10)
    assert ledger.advanced() == {PARTITION: 14}


def test_ledger_counts_a_redelivered_offset_twice() -> None:
    ledger = ConcurrentCommitLedger()
    ledger.dispatched(PARTITION, 10)
    ledger.dispatched(PARTITION, 10)
    ledger.finished(PARTITION, 10)
    assert ledger.position(PARTITION) == 10, "the second delivery still runs"
    ledger.finished(PARTITION, 10)
    assert ledger.position(PARTITION) == 11


def test_ledger_respects_a_rewind_floor_and_rebases() -> None:
    ledger = ConcurrentCommitLedger()
    for offset in (10, 11, 12):
        ledger.dispatched(PARTITION, offset)
        ledger.finished(PARTITION, offset)
    assert ledger.advanced(rewind_floors={PARTITION: 11}) == {PARTITION: 11}
    ledger.rebase(PARTITION, 11)
    assert ledger.position(PARTITION) == 11


# --------------------------------------------------------------------------
# 2. A declared-concurrent subscription does not auto-commit.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declared_concurrent_consumer_is_built_without_auto_commit() -> None:
    bus = EventBusKafka(config=_config(enable_auto_commit=True))
    bus.declare_consume_concurrency(
        topic=TOPIC, group_id=GROUP, max_in_flight_records=8
    )
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer"
    ) as consumer_cls:
        bus._build_consumer(
            TOPIC, f"{GROUP}.__t.{TOPIC}", "inst", "latest", group_id=GROUP
        )
        bus._build_consumer(
            "serial-topic", "serial.__t.x", "inst", "latest", group_id="serial"
        )
    concurrent_kwargs = consumer_cls.call_args_list[0].kwargs
    serial_kwargs = consumer_cls.call_args_list[1].kwargs
    assert concurrent_kwargs["enable_auto_commit"] is False, (
        "OMN-20117: auto-commit commits the fetch position, which is past every "
        "record still in flight on a concurrent subscription; a restart then "
        "resumes past them and they are never redelivered"
    )
    assert serial_kwargs["enable_auto_commit"] is True, "the serial path is unchanged"


# --------------------------------------------------------------------------
# 3. The concurrent driver commits a low watermark.
# --------------------------------------------------------------------------


async def _start_concurrent_loop(
    bus: EventBusKafka,
    consumer: _FakeConsumer,
    dispatch: Any,
) -> asyncio.Task[None]:
    bus._group_consumers[(TOPIC, GROUP)] = consumer  # type: ignore[assignment]
    bus.declare_consume_concurrency(
        topic=TOPIC, group_id=GROUP, max_in_flight_records=4
    )

    async def _subscriber(_message: Any) -> None:
        """Present so the record has a subscriber to reach."""

    bus._subscribers[TOPIC].append((GROUP, "omn20117-subscription", _subscriber))
    patcher = patch.object(bus, "_dispatch_to_subscriber", side_effect=dispatch)
    patcher.start()
    task = asyncio.create_task(bus._consume_loop(TOPIC, GROUP, uuid4()))
    task.add_done_callback(lambda _t: patcher.stop())
    bus._group_consumer_tasks[(TOPIC, GROUP)] = task
    return task


@pytest.mark.asyncio
async def test_commit_is_never_past_a_record_still_in_flight() -> None:
    running: set[int] = set()
    positions_while_running: list[tuple[int, set[int]]] = []
    consumer = _FakeConsumer([_msg(BASE + i) for i in range(4)], batch_size=4)
    original_commit = consumer.commit

    async def _recording_commit(offsets: dict[TopicPartition, int]) -> None:
        for value in offsets.values():
            positions_while_running.append(
                (int(getattr(value, "offset", value)), set(running))
            )
        await original_commit(offsets)

    consumer.commit = _recording_commit  # type: ignore[method-assign]

    async def _dispatch(*args: Any, record_coordinate: Any = None, **_: Any) -> bool:
        offset = int(record_coordinate[1])
        running.add(offset)
        try:
            await asyncio.sleep(0.4 if offset == BASE else 0.02)
        finally:
            running.discard(offset)
        return True

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()
        await _start_concurrent_loop(bus, consumer, _dispatch)
        await asyncio.sleep(0.8)
        await bus.close()

    assert positions_while_running, (
        "OMN-20117: the concurrent driver never committed. Under auto-commit the "
        "client commits the fetch position, past every record still running, "
        "so a restart resumes past them"
    )
    for position, live in positions_while_running:
        if live:
            assert position <= min(live), (
                f"committed {position} while {sorted(live)} were still running"
            )
    assert max(consumer.committed_offsets()) == BASE + 4


# --------------------------------------------------------------------------
# 4. close() lets running dispatches publish; cancels the rest.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_lets_a_running_dispatch_publish_its_terminal() -> None:
    published: list[str] = []
    errors: list[BaseException] = []
    consumer = _FakeConsumer([_msg(BASE)], batch_size=1)

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config(consumer_shutdown_drain_seconds=2.0))
        await bus.start()

        async def _dispatch(
            *args: Any, record_coordinate: Any = None, **_: Any
        ) -> bool:
            await asyncio.sleep(0.3)
            try:
                await bus.publish("onex.evt.omn20117.completed.v1", None, b"done")
                published.append("completed")
            except Exception as exc:  # noqa: BLE001 - the assertion reads it
                errors.append(exc)
            return True

        await _start_concurrent_loop(bus, consumer, _dispatch)
        await asyncio.sleep(0.1)
        await bus.close()

    assert not errors, (
        "OMN-20117: close() refused the terminal of a handler that finished "
        f"during shutdown: {errors!r}. On the dev lane six delegate-skill "
        "terminals were lost this way and routed to the DLQ"
    )
    assert published == ["completed"]
    assert max(consumer.committed_offsets()) == BASE + 1


@pytest.mark.asyncio
async def test_cancelled_dispatch_can_answer_and_is_committed() -> None:
    published: list[str] = []
    consumer = _FakeConsumer([_msg(BASE)], batch_size=1)

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(
            config=_config(
                consumer_shutdown_drain_seconds=0.2,
                consumer_shutdown_cancel_grace_seconds=2.0,
            )
        )
        await bus.start()

        async def _handler() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                # A cancellation-aware handler answers with a failure terminal.
                await bus.publish("onex.evt.omn20117.failed.v1", None, b"shutdown")
                published.append("failed")

        async def _dispatch(
            *args: Any, record_coordinate: Any = None, **kwargs: Any
        ) -> bool:
            # Go through the real deadline wrapper so the bus can see and
            # cancel the inner dispatch future.
            await bus._await_dispatch_within_deadline(
                lambda _m: _handler(),
                None,
                topic=TOPIC,
                group_id=GROUP,
                subscription_id="omn20117-subscription",
                correlation_id=uuid4(),
                record_coordinate=record_coordinate,
                on_slow_dispatch=None,
            )
            return True

        await _start_concurrent_loop(bus, consumer, _dispatch)
        await asyncio.sleep(0.1)
        await bus.close()

    assert published == ["failed"], (
        "OMN-20117: a handler still running at the end of the drain must be "
        "cancelled while the producer is open, so it can tell its caller"
    )
    assert max(consumer.committed_offsets()) == BASE + 1


@pytest.mark.asyncio
async def test_cancelled_dispatch_that_does_not_answer_is_not_committed() -> None:
    consumer = _FakeConsumer([_msg(BASE), _msg(BASE + 1)], batch_size=2)

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(
            config=_config(
                consumer_shutdown_drain_seconds=0.2,
                consumer_shutdown_cancel_grace_seconds=0.5,
            )
        )
        await bus.start()

        async def _dispatch(
            *args: Any, record_coordinate: Any = None, **_: Any
        ) -> bool:
            offset = int(record_coordinate[1])

            async def _handler() -> None:
                await asyncio.sleep(60 if offset == BASE else 0.01)

            await bus._await_dispatch_within_deadline(
                lambda _m: _handler(),
                None,
                topic=TOPIC,
                group_id=GROUP,
                subscription_id="omn20117-subscription",
                correlation_id=uuid4(),
                record_coordinate=record_coordinate,
                on_slow_dispatch=None,
            )
            return True

        await _start_concurrent_loop(bus, consumer, _dispatch)
        await asyncio.sleep(0.1)
        await bus.close()

    assert all(offset <= BASE for offset in consumer.committed_offsets()), (
        "OMN-20117: a record whose handler never finished was committed past, "
        f"so the next consumer would skip it: {consumer.committed_offsets()}"
    )


# --------------------------------------------------------------------------
# 5. The serial path pins the position when shutdown cuts a batch short.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serial_batch_cut_by_shutdown_pins_the_fetch_position() -> None:
    consumer = _FakeConsumer([_msg(BASE + i) for i in range(3)], batch_size=3)
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=_producer(),
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()

        async def _process(msg: Any, *args: Any, **kwargs: Any) -> bool:
            bus._shutdown = True  # the SIGTERM lands while the first record runs
            return False

        with patch.object(bus, "_process_consumed_record", side_effect=_process):
            await bus._process_serial_batch(
                {TopicPartition(TOPIC, PARTITION): [_msg(BASE + i) for i in range(3)]},
                topic=TOPIC,
                group_id=GROUP,
                correlation_id=uuid4(),
                consumer=consumer,  # type: ignore[arg-type]
            )
        bus._shutdown = False
        await bus.close()

    assert (TopicPartition(TOPIC, PARTITION), BASE + 1) in consumer.seeks, (
        "OMN-20117: shutdown returned from the batch without pinning the fetch "
        "position, so auto-commit commits past the two records never processed"
    )

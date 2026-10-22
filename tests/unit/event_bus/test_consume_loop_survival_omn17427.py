# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: while the bus runs, every registered subscription has a polling member.

THE INCIDENT. On the dev lane on 2026-10-22 one consumer group of the
effects runtime sat ``Empty`` with 0 members while its two siblings on the same
topic were ``Stable``, and the runtime's own ``consumer_sync`` dimension read
HEALTHY. aiokafka keeps a member's group through heartbeats for
``max_poll_interval_ms`` after its owner stops calling ``getmany``, then leaves
and never rejoins on its own: a rejoin only happens inside the next poll. A
consume loop that had ended on an error, or was parked in an await, had lost
its group for good, and nothing in the process noticed.

WHAT THESE TESTS HOLD THE BUS TO.

* A poll or a batch that raises -- a ``KafkaError``, a plain ``RuntimeError``,
  or a ``CancelledError`` raised out of a subscriber's own coroutine -- does not
  end the loop. The consumer is replaced and the loop polls on.
* Cancelling the loop task itself still ends it promptly, and close() still
  closes the consumer.
* A consume task that has ended, or that has not polled for longer than the
  parked bound, is restarted by the watchdog with a rebuilt consumer, the
  restart is recorded as a typed rejoin event, and the cooldown bounds it.
* ``consumer_sync`` and ``get_readiness_status`` see a group whose loop stopped.

No broker: fake consumers, an injectable clock and sub-second intervals.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.errors import KafkaConnectionError
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.event_bus.consumer_rejoin_policy import ModelConsumerRejoinPolicy
from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
    ConsumerRejoinSupervisor,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer

    from omnibase_infra.event_bus.models import ModelEventMessage

    _Callback = Callable[[ModelEventMessage], Awaitable[None]]

pytestmark = pytest.mark.unit

TOPIC = "onex.evt.platform.node-heartbeat.v1"
GROUP = "local.omnimarket.topic_activity_sampler_effect.consume.1.0.0"
SIBLING_GROUP = "local.omnimarket.node_heartbeat_projection.consume.1.0.0"
PARTITION = 0
TP = TopicPartition(TOPIC, PARTITION)

#: Bound on every drive of the real loop. Before OMN-17427 a raised poll ended
#: the loop at once, so the red run fails on its assertions, not on this bound.
LOOP_BOUND_SECONDS = 5.0


class _FakeClock:
    """Monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _PartitionLog:
    """``AIOKafkaConsumer`` stand-in over one partition with a real fetch position.

    ``failures`` are raised by the next polls, one per poll, before any record
    is returned -- the shape of a broker error surfacing through ``getmany``.
    """

    def __init__(
        self,
        offsets: list[int],
        *,
        failures: list[BaseException] | None = None,
        block: bool = False,
    ) -> None:
        self._records = [_raw_msg(offset) for offset in offsets]
        self._failures = list(failures or [])
        self._block = block
        self.fetch_position = offsets[0] if offsets else 0
        self.seeks: list[int] = []
        self.polls = 0
        self.stopped = False
        self.on_drained: Callable[[], None] | None = None

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[Any]]:
        self.polls += 1
        await asyncio.sleep(0)
        if self._block:
            await asyncio.Event().wait()
        if self._failures:
            raise self._failures.pop(0)
        batch = [r for r in self._records if r.offset >= self.fetch_position]
        if not batch:
            if self.on_drained is not None:
                self.on_drained()
            await asyncio.sleep(0.005)
            return {}
        self.fetch_position = batch[-1].offset + 1
        return {TP: batch}

    def assignment(self) -> set[TopicPartition]:
        return {TP}

    async def position(self, partition: TopicPartition) -> int:
        return self.fetch_position

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {TP: self.fetch_position}

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seeks.append(offset)
        self.fetch_position = offset

    async def commit(self, offsets: Any = None) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True


def _raw_msg(offset: int) -> MagicMock:
    msg = MagicMock()
    msg.topic = TOPIC
    msg.partition = PARTITION
    msg.offset = offset
    msg.timestamp = int(datetime.now(UTC).timestamp() * 1000)
    msg.key = None
    msg.headers = []
    msg.value = b"{}"
    return msg


def _config(**overrides: Any) -> ModelKafkaEventBusConfig:
    values: dict[str, Any] = {
        "bootstrap_servers": "localhost:9092",
        "max_poll_interval_ms": 10_000,
        "session_timeout_ms": 6_000,
        "heartbeat_interval_ms": 2_000,
        "consumer_poll_timeout_ms": 100,
        "consumer_dispatch_deadline_seconds": 2.0,
        "consume_loop_failure_backoff_initial_seconds": 0.01,
        "consume_loop_failure_backoff_max_seconds": 0.05,
        "consume_task_watchdog_interval_seconds": 0.05,
    }
    values.update(overrides)
    return ModelKafkaEventBusConfig(**values)


class _Harness:
    """The real ``EventBusKafka`` consume loop over fake consumers.

    ``_rebuild_consumer_for_rejoin`` is replaced by one that hands out the
    next consumer from ``replacements`` and publishes it the way the real one
    does, so every replacement the bus orders is counted.
    """

    def __init__(
        self,
        config: ModelKafkaEventBusConfig,
        first: _PartitionLog,
        replacements: list[_PartitionLog],
        *,
        handler: Callable[[int], Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.config = config
        self.first = first
        self.replacements = list(replacements)
        self.rebuilt: list[_PartitionLog] = []
        self.handled: list[int] = []
        self.dlq_calls: list[dict[str, Any]] = []
        self._handler = handler
        self._max_retries = max_retries
        self.bus: EventBusKafka | None = None
        self._patches: list[Any] = []

    async def __aenter__(self) -> _Harness:
        producer = AsyncMock()
        producer._closed = False
        producer_patch = patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=producer,
        )
        producer_patch.start()
        self._patches.append(producer_patch)
        bus = EventBusKafka(config=self.config)
        self.bus = bus
        await bus.start()
        bus._group_consumers[(TOPIC, GROUP)] = cast("AIOKafkaConsumer", self.first)
        self._wire_drain(self.first)

        async def _callback(message: Any) -> None:
            if self._handler is not None:
                await self._handler(message.raw_offset)
            self.handled.append(message.raw_offset)

        bus._subscribers[TOPIC] = [(GROUP, "sub-1", cast("_Callback", _callback))]

        def _to_model(msg: Any, _topic: str) -> MagicMock:
            message = MagicMock()
            message.raw_offset = msg.offset
            message.headers.retry_count = 0
            message.headers.max_retries = self._max_retries
            return message

        async def _record_dlq(**kwargs: Any) -> bool:
            self.dlq_calls.append(kwargs)
            return True

        async def _rebuild(topic: str, group_id: str) -> Any:
            replacement = self.replacements.pop(0)
            self._wire_drain(replacement)
            self.rebuilt.append(replacement)
            bus._group_consumers[(topic, group_id)] = cast(
                "AIOKafkaConsumer", replacement
            )
            return replacement

        for active in (
            patch.object(bus, "_kafka_msg_to_model", side_effect=_to_model),
            patch.object(bus, "_publish_to_dlq", side_effect=_record_dlq),
            patch.object(bus, "_rebuild_consumer_for_rejoin", side_effect=_rebuild),
            patch.object(
                event_bus_kafka_module, "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS", 0.0
            ),
        ):
            active.start()
            self._patches.append(active)
        return self

    def _wire_drain(self, log: _PartitionLog) -> None:
        bus = self.bus
        assert bus is not None
        if log._records:
            log.on_drained = lambda: setattr(bus, "_shutdown", True)

    async def run_loop(self) -> None:
        assert self.bus is not None
        await asyncio.wait_for(
            self.bus._consume_loop(TOPIC, GROUP, uuid4()),
            timeout=LOOP_BOUND_SECONDS,
        )

    async def __aexit__(self, *exc: object) -> None:
        if self.bus is not None:
            await self.bus.close()
        for active in reversed(self._patches):
            active.stop()


# ---------------------------------------------------------------------------
# (1) Loop survival
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(KafkaConnectionError("coordinator went away"), id="kafka_error"),
        pytest.param(RuntimeError("fetcher blew up"), id="runtime_error"),
    ],
)
@pytest.mark.parametrize("concurrency", [1, 2], ids=["serial", "concurrent"])
async def test_a_poll_that_raises_once_rebuilds_the_consumer_and_keeps_consuming(
    failure: BaseException, concurrency: int
) -> None:
    """Falsifier: before OMN-17427 the loop ENDED after the first raise."""
    broken = _PartitionLog([], failures=[failure])
    replacement = _PartitionLog([10, 11, 12])
    async with _Harness(_config(), broken, [replacement]) as harness:
        assert harness.bus is not None
        if concurrency > 1:
            harness.bus.declare_consume_concurrency(
                topic=TOPIC, group_id=GROUP, max_in_flight_records=concurrency
            )
        await harness.run_loop()

        assert harness.rebuilt == [replacement], (
            "the consume loop did not replace its consumer after a failed poll "
            "-- it ended, and the group loses its member once "
            "max_poll_interval_ms passes with nothing polling (OMN-17427)"
        )
        assert broken.stopped, "the failed consumer must be stopped before replacement"
        assert sorted(harness.handled) == [10, 11, 12]


@pytest.mark.asyncio
async def test_a_batch_that_raises_rebuilds_from_the_batch_start() -> None:
    """A batch whose processing raises is refetched, never skipped."""
    log = _PartitionLog([20, 21, 22])
    replacement = _PartitionLog([20, 21, 22])
    async with _Harness(_config(), log, [replacement]) as harness:
        assert harness.bus is not None
        real = harness.bus._process_serial_batch
        calls = {"n": 0}

        async def _flaky(*args: Any, **kwargs: Any) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("batch infrastructure failed")
            await real(*args, **kwargs)

        with patch.object(harness.bus, "_process_serial_batch", side_effect=_flaky):
            await harness.run_loop()

        assert log.seeks == [20], (
            "the failed batch's fetch position must be pinned at its first "
            "record before the consumer is stopped, or stop() auto-commits "
            "past records that never ran"
        )
        assert harness.rebuilt == [replacement]
        assert harness.handled == [20, 21, 22]


@pytest.mark.asyncio
async def test_a_subscriber_raising_cancelled_error_takes_the_handler_failure_path() -> (
    None
):
    """Falsifier: before OMN-17427 the CancelledError ended the consume task."""

    async def _handler(offset: int) -> None:
        if offset == 30:
            raise asyncio.CancelledError

    log = _PartitionLog([30, 31, 32])
    async with _Harness(_config(), log, [], handler=_handler, max_retries=0) as h:
        await h.run_loop()

        assert h.handled == [31, 32], (
            "a CancelledError raised inside a subscriber must not end the loop; "
            "the next record must be dispatched"
        )
        assert len(h.dlq_calls) == 1, "the cancelled record takes the DLQ arm"
        assert type(h.dlq_calls[0]["error"]).__name__ == "DispatchCancelledError"
        assert h.rebuilt == [], "a handler failure is not a consumer failure"


@pytest.mark.asyncio
async def test_cancelling_the_loop_task_ends_it_promptly_and_close_stops_the_consumer() -> (
    None
):
    """Shutdown still works, including while the loop is between failures."""
    failing = _PartitionLog(
        [], failures=[RuntimeError(f"down {i}") for i in range(100)]
    )
    config = _config(
        consume_loop_failure_backoff_initial_seconds=0.5,
        consume_loop_failure_backoff_max_seconds=0.5,
    )
    async with _Harness(config, failing, [failing] * 10) as harness:
        bus = harness.bus
        assert bus is not None
        task = asyncio.create_task(bus._consume_loop(TOPIC, GROUP, uuid4()))
        bus._group_consumer_tasks[(TOPIC, GROUP)] = task
        await asyncio.sleep(0.05)  # inside the first backoff
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        blocked = _PartitionLog([], block=True)
        bus._group_consumers[(TOPIC, GROUP)] = cast("AIOKafkaConsumer", blocked)
        task = asyncio.create_task(bus._consume_loop(TOPIC, GROUP, uuid4()))
        bus._group_consumer_tasks[(TOPIC, GROUP)] = task
        await asyncio.sleep(0.02)
        await asyncio.wait_for(bus.close(), timeout=2.0)
        assert task.done()
        assert blocked.stopped


# ---------------------------------------------------------------------------
# (2) Consume-task watchdog
# ---------------------------------------------------------------------------


async def _finished_task() -> asyncio.Task[None]:
    async def _ended() -> None:
        return None

    task = asyncio.create_task(_ended())
    await task
    return task


async def _stop(bus: EventBusKafka) -> None:
    bus._shutdown = True
    for task in list(bus._group_consumer_tasks.values()):
        task.cancel()
        with pytest.raises((asyncio.CancelledError, TimeoutError)):
            await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_the_watchdog_restarts_a_dead_consume_task_within_one_interval() -> None:
    """Falsifier: before OMN-17427 nothing ever read a dead task back."""
    idle = _PartitionLog([])
    replacement = _PartitionLog([])
    async with _Harness(_config(), idle, [replacement]) as harness:
        bus = harness.bus
        assert bus is not None
        dead = await _finished_task()
        bus._group_consumer_tasks[(TOPIC, GROUP)] = dead

        bus._ensure_consume_watchdog()
        for _ in range(40):  # at most 20 watchdog intervals of 0.05s
            await asyncio.sleep(0.025)
            if bus._group_consumer_tasks[(TOPIC, GROUP)] is not dead:
                break

        restarted = bus._group_consumer_tasks[(TOPIC, GROUP)]
        assert restarted is not dead, "the dead consume task was never restarted"
        assert not restarted.done()
        assert harness.rebuilt == [replacement]
        assert idle.stopped
        events = bus.consumer_rejoin_events()
        assert [e.reason for e in events] == [EnumConsumerStallReason.CONSUME_TASK_DEAD]
        assert events[0].rejoin_succeeded is True
        await _stop(bus)


@pytest.mark.asyncio
async def test_the_watchdog_cancels_and_restarts_a_parked_consume_task() -> None:
    """A loop that stopped polling while alive is parked, and replaced."""
    clock = _FakeClock()
    first = _PartitionLog([])
    replacement = _PartitionLog([])
    async with _Harness(_config(), first, [replacement]) as harness:
        bus = harness.bus
        assert bus is not None
        bus._clock = clock
        supervisor = bus._rejoin_supervisor_for(TOPIC, GROUP)
        await supervisor.next_batch(first)  # one poll, at clock.now

        parked = asyncio.create_task(asyncio.Event().wait())
        bus._group_consumer_tasks[(TOPIC, GROUP)] = parked

        bound = bus._config.effective_consume_task_parked_after_seconds
        clock.now += bound - 1.0
        await bus._check_consume_tasks()
        assert bus._group_consumer_tasks[(TOPIC, GROUP)] is parked, (
            "a loop inside the parked bound must be left alone"
        )

        clock.now += 2.0
        await bus._check_consume_tasks()
        assert parked.cancelled(), "the parked consume task must be cancelled"
        restarted = bus._group_consumer_tasks[(TOPIC, GROUP)]
        assert restarted is not parked and not restarted.done()
        assert harness.rebuilt == [replacement]
        assert [e.reason for e in bus.consumer_rejoin_events()] == [
            EnumConsumerStallReason.CONSUME_TASK_PARKED
        ]
        await _stop(bus)


@pytest.mark.asyncio
async def test_a_watchdog_restart_inside_the_cooldown_does_not_happen_twice() -> None:
    clock = _FakeClock()
    first = _PartitionLog([])
    replacements = [_PartitionLog([]), _PartitionLog([])]
    config = _config(consumer_rejoin_cooldown_seconds=60.0)
    async with _Harness(config, first, list(replacements)) as harness:
        bus = harness.bus
        assert bus is not None
        bus._clock = clock

        bus._group_consumer_tasks[(TOPIC, GROUP)] = await _finished_task()
        await bus._check_consume_tasks()
        assert len(harness.rebuilt) == 1

        # The restarted task dies again at once: inside the cooldown, no restart.
        restarted = bus._group_consumer_tasks[(TOPIC, GROUP)]
        restarted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await restarted
        clock.now += 30.0
        await bus._check_consume_tasks()
        assert len(harness.rebuilt) == 1, "a restart inside the cooldown happened"
        assert bus._group_consumer_tasks[(TOPIC, GROUP)] is restarted

        clock.now += 31.0
        await bus._check_consume_tasks()
        assert len(harness.rebuilt) == 2, "past the cooldown the restart must retry"
        await _stop(bus)


@pytest.mark.asyncio
async def test_the_watchdog_leaves_an_unsubscribed_group_alone() -> None:
    first = _PartitionLog([])
    async with _Harness(_config(), first, [_PartitionLog([])]) as harness:
        bus = harness.bus
        assert bus is not None
        bus._subscribers[TOPIC] = []
        dead = await _finished_task()
        bus._group_consumer_tasks[(TOPIC, GROUP)] = dead
        await bus._check_consume_tasks()
        assert bus._group_consumer_tasks[(TOPIC, GROUP)] is dead
        assert harness.rebuilt == []


# ---------------------------------------------------------------------------
# (3) Readiness sees it
# ---------------------------------------------------------------------------


def _supervisor(
    clock: _FakeClock, *, unready: float = 600.0
) -> ConsumerRejoinSupervisor:
    async def _recreate() -> Any:
        raise AssertionError("no rejoin expected")

    return ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=ModelConsumerRejoinPolicy(
            stall_seconds=120.0,
            required_consecutive_stalls=3,
            rejoin_cooldown_seconds=300.0,
            sync_unready_seconds=unready,
        ),
        poll_timeout_ms=100,
        recreate_consumer=_recreate,
        clock=clock,
    )


@pytest.mark.asyncio
async def test_sync_status_reports_a_group_whose_last_poll_is_stale_as_stalled() -> (
    None
):
    """Falsifier: before OMN-17427 sync_status replayed the last evaluation."""
    clock = _FakeClock()
    supervisor = _supervisor(clock)
    await supervisor.next_batch(_PartitionLog([1]))
    fresh = supervisor.sync_status()
    assert fresh.ready and fresh.stalled_seconds == 0.0

    clock.now += 601.0
    stale = supervisor.sync_status()
    assert stale.evaluated is True
    assert stale.reason is EnumConsumerStallReason.CONSUME_TASK_PARKED
    assert stale.stalled_seconds == pytest.approx(601.0)
    assert stale.ready is False


@pytest.mark.asyncio
async def test_sync_status_of_a_recently_polled_group_is_unchanged() -> None:
    clock = _FakeClock()
    supervisor = _supervisor(clock)
    await supervisor.next_batch(_PartitionLog([1]))
    clock.now += 500.0
    await supervisor.next_batch(_PartitionLog([]))
    clock.now += 10.0
    status = supervisor.sync_status()
    assert status.reason is not EnumConsumerStallReason.CONSUME_TASK_PARKED
    assert status.ready is True


@pytest.mark.asyncio
async def test_readiness_reports_a_topic_dead_when_any_of_its_groups_is_dead() -> None:
    """Falsifier: before OMN-17427 one live sibling hid the dead group."""
    async with _Harness(_config(), _PartitionLog([]), []) as harness:
        bus = harness.bus
        assert bus is not None
        bus._group_consumers[(TOPIC, SIBLING_GROUP)] = cast(
            "AIOKafkaConsumer", _PartitionLog([])
        )
        alive = asyncio.create_task(asyncio.Event().wait())
        bus._group_consumer_tasks[(TOPIC, SIBLING_GROUP)] = alive
        bus._group_consumer_tasks[(TOPIC, GROUP)] = await _finished_task()

        status = await bus.get_readiness_status()
        assert status.consume_tasks_alive[TOPIC] is False

        alive.cancel()
        with pytest.raises(asyncio.CancelledError):
            await alive

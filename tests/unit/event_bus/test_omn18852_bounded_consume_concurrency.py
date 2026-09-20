# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18852: a consume path that declares it must hold N records in flight.

Measured defect, ``.201`` dev lane, 2026-09-19. Every LLM inference on the
lane -- local AND cloud -- is served by one partition with one consumer-group
member, and ``EventBusKafka._consume_loop`` awaits the handler inline inside
the poll loop::

    for records in batch.records.values():
        for msg in records:
            rewound = await self._process_consumed_record(...)

so the lane executes exactly one inference at a time, globally. Fifteen
consecutive calls spanning eight correlations and three providers produced
**zero overlapping pairs**; queue wait grew monotonically from 3 s to 445 s
across nine delegations in 26 minutes, and three callers timed out at the
CLI's ~306 s while their answers were produced and published after they had
exited. A control run spent 179 s of its 181 s wall clock queued behind an
inference that took 1.559 s.

The headline test here is ``test_declared_concurrency_overlaps_in_time``: it
asserts records OVERLAP, by comparing start/end windows, rather than merely
that they all completed. "All N completed" passes on the serial path and is
the assertion that would have let this ship.

Three properties are pinned alongside it, because each is a way the fix could
be worse than the defect:

* **The undeclared path is unchanged.** Not "a semaphore of size one" -- the
  same inline ``await``, proven by asserting the rewind seek is issued from
  inside record processing rather than deferred, and that no dispatch task is
  spawned.
* **A rewind is fail-closed under concurrency.** A rewind issued by one
  in-flight task would otherwise replay records a sibling already completed,
  and concurrent ``seek()`` calls on one TopicPartition race. The loop drains
  before seeking and seeks to the LOWEST offset that asked for one, so nothing
  after the rewind point is skipped.
* **Ordering is given up deliberately.** A topic that opts in no longer
  preserves global partition ordering. Delegations and inference intents are
  independent per correlation id, so only per-key ordering is required.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_TOPIC: str = "omn18852-inference-request"
TEST_GROUP: str = "test.omnimarket.omn18852.consume.v1"
TEST_PARTITION: int = 0
BASE_OFFSET: int = 1000

#: Long enough that a serial run is unambiguously longer than a concurrent
#: one on a loaded CI box, short enough that the file stays a unit test.
HANDLER_SECONDS: float = 0.20

pytestmark = pytest.mark.unit


class _Interval:
    """One handler invocation's wall-clock window."""

    __slots__ = ("finished", "offset", "started")

    def __init__(self, offset: int, started: float) -> None:
        self.offset = offset
        self.started = started
        self.finished = float("inf")

    def overlaps(self, other: _Interval) -> bool:
        return self.started < other.finished and other.started < self.finished


def count_overlapping_pairs(intervals: Sequence[_Interval]) -> int:
    """Pairs whose wall-clock windows intersect.

    This is the measurement from the diagnosis, reproduced as an assertion:
    the live lane's fifteen inferences produced zero of these.
    """
    return sum(
        1
        for i, left in enumerate(intervals)
        for right in intervals[i + 1 :]
        if left.overlaps(right)
    )


class _FakeConsumer:
    """Async stand-in for ``AIOKafkaConsumer`` that records ``seek`` calls.

    Modelled on the OMN-15232 fixture. ``batch_size`` controls how many
    records one ``getmany`` returns, because the whole question is what the
    loop does with a batch it has already fetched.
    """

    def __init__(self, messages: list[Any], *, batch_size: int) -> None:
        self._messages = list(messages)
        self._batch_size = batch_size
        self.seek_calls: list[tuple[TopicPartition, int]] = []
        self.seek_observed_in_flight: list[int] = []
        self.on_drained: Callable[[], None] | None = None
        self.in_flight_probe: Callable[[], int] | None = None

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[Any]]:
        if not self._messages:
            if self.on_drained is not None:
                self.on_drained()
            return {}
        batch = self._messages[: self._batch_size]
        del self._messages[: self._batch_size]
        return {TopicPartition(TEST_TOPIC, TEST_PARTITION): batch}

    def assignment(self) -> set[TopicPartition]:
        return set()

    def seek(self, topic_partition: TopicPartition, offset: int) -> None:
        self.seek_calls.append((topic_partition, offset))
        # How many handlers were still running when the seek was issued. A
        # concurrent seek against a live sibling is the race this change has
        # to avoid, so the fixture measures it rather than assuming it.
        if self.in_flight_probe is not None:
            self.seek_observed_in_flight.append(self.in_flight_probe())

    async def stop(self) -> None:
        """No-op; the loop's cleanup path calls this on shutdown."""


def _make_msg(offset: int) -> MagicMock:
    msg = MagicMock()
    msg.topic = TEST_TOPIC
    msg.partition = TEST_PARTITION
    msg.offset = offset
    msg.key = f"omn18852-{offset}".encode()
    msg.value = b'{"payload": "ok"}'
    msg.headers = ()
    return msg


@pytest.fixture
def bus_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092",
        environment="dev",
        dead_letter_topic="dlq-events",
    )


@pytest.fixture
def mock_producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    producer._closed = False
    return producer


class _Harness:
    """One driven ``_consume_loop`` run and everything it recorded."""

    def __init__(self) -> None:
        self.intervals: list[_Interval] = []
        self.live: int = 0
        self.peak_live: int = 0
        self.delivered_offsets: list[int] = []
        self.spawned_tasks: int = 0
        self.consumer: _FakeConsumer | None = None
        self.elapsed: float = 0.0


async def _drive_consume_loop(
    bus_config: ModelKafkaEventBusConfig,
    mock_producer: AsyncMock,
    *,
    record_count: int,
    batch_size: int,
    declared_concurrency: int | None,
    handler_seconds: float = HANDLER_SECONDS,
    dlq_unpersisted_offsets: frozenset[int] = frozenset(),
) -> _Harness:
    """Run the REAL ``_consume_loop`` over ``record_count`` records.

    ``dlq_unpersisted_offsets`` names records whose handler fails with retries
    exhausted AND whose DLQ write is not confirmed -- the OMN-15232 condition
    that forces a fail-closed rewind.
    """
    harness = _Harness()
    offsets = [BASE_OFFSET + i for i in range(record_count)]

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        event_bus = EventBusKafka(config=bus_config)
        await event_bus.start()

        consumer = _FakeConsumer([_make_msg(o) for o in offsets], batch_size=batch_size)
        harness.consumer = consumer
        event_bus._group_consumers[(TEST_TOPIC, TEST_GROUP)] = consumer  # type: ignore[assignment]
        consumer.on_drained = lambda: setattr(event_bus, "_shutdown", True)
        consumer.in_flight_probe = lambda: harness.live

        if declared_concurrency is not None:
            event_bus.declare_consume_concurrency(
                topic=TEST_TOPIC,
                group_id=TEST_GROUP,
                max_in_flight_records=declared_concurrency,
            )

        # The subscriber is the thing whose wall-clock windows we measure.
        # ``_dispatch_to_subscriber`` is patched rather than the callback so
        # the DLQ/offset-safety return value is under the test's control,
        # which is what the rewind cases need.
        async def _dispatch(
            _callback: Any,
            _subscription_id: str,
            event_message: Any,
            _topic: str,
            _group_id: str,
            _correlation_id: Any,
            *,
            record_coordinate: Any = None,
        ) -> bool:
            # ``_record_coordinate`` yields a ``(partition, offset)`` tuple.
            assert record_coordinate is not None, (
                "the loop must pass the record coordinate through to dispatch"
            )
            offset = int(record_coordinate[1])
            interval = _Interval(offset, time.monotonic())
            harness.intervals.append(interval)
            harness.live += 1
            harness.peak_live = max(harness.peak_live, harness.live)
            try:
                await asyncio.sleep(handler_seconds)
                harness.delivered_offsets.append(offset)
                return offset not in dlq_unpersisted_offsets
            finally:
                interval.finished = time.monotonic()
                harness.live -= 1

        real_create_task = asyncio.create_task

        def _counting_create_task(coro: Any, **kwargs: Any) -> Any:
            harness.spawned_tasks += 1
            return real_create_task(coro, **kwargs)

        async def _one_subscriber(_message: Any) -> None:
            """Registered so ``_process_consumed_record`` has a subscriber."""

        event_bus._subscribers[TEST_TOPIC].append(
            (TEST_GROUP, "omn18852-subscription", _one_subscriber)
        )

        started = time.monotonic()
        with (
            patch.object(event_bus, "_dispatch_to_subscriber", side_effect=_dispatch),
            patch.object(
                event_bus_kafka_module,
                "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS",
                0.0,
                create=True,
            ),
            patch.object(
                event_bus_kafka_module.asyncio,
                "create_task",
                side_effect=_counting_create_task,
            ),
        ):
            await event_bus._consume_loop(TEST_TOPIC, TEST_GROUP, uuid4())
        harness.elapsed = time.monotonic() - started

        await event_bus.close()

    return harness


# --------------------------------------------------------------------------
# AC1 headline: records declared concurrent must OVERLAP IN TIME.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declared_concurrency_overlaps_in_time(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """RED against dev: four declared-concurrent records must run at once.

    The live lane's failure signature is the zero in ``count_overlapping_pairs``
    -- fifteen inferences, three providers, not one overlapping pair. Asserting
    only that all four completed passes on the serial path, which is why this
    compares start/end windows instead.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=4,
        batch_size=4,
        declared_concurrency=4,
    )

    assert len(harness.intervals) == 4, "every record must reach a subscriber"

    overlaps = count_overlapping_pairs(harness.intervals)
    assert overlaps > 0, (
        "OMN-18852: a consume path declaring max_in_flight_records=4 executed "
        f"its records with {overlaps} overlapping pairs -- i.e. serially. This "
        "is the exact measurement taken on the .201 dev lane on 2026-09-19: "
        "15 consecutive inferences across 3 providers, ZERO overlapping pairs, "
        "queue wait growing 3s -> 445s. The handler is awaited inline inside "
        "the poll loop, so declaring a bound changed nothing."
    )
    assert harness.peak_live >= 2, (
        f"peak concurrent handlers was {harness.peak_live}; a declared bound "
        "of 4 must hold more than one record in flight"
    )
    serial_floor = 4 * HANDLER_SECONDS
    assert harness.elapsed < serial_floor, (
        f"elapsed {harness.elapsed:.3f}s is at or above the serial floor "
        f"{serial_floor:.3f}s, so the records did not actually run concurrently"
    )


@pytest.mark.asyncio
async def test_declared_bound_is_an_upper_bound_not_a_target(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """Six records under a bound of 2 must never have three in flight."""
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=6,
        batch_size=6,
        declared_concurrency=2,
        handler_seconds=0.05,
    )

    assert len(harness.intervals) == 6
    assert harness.peak_live == 2, (
        f"peak concurrent handlers was {harness.peak_live}, but the declared "
        "bound was 2; an unbounded dispatch converts a queueing problem into a "
        "resource-exhaustion one"
    )


@pytest.mark.asyncio
async def test_concurrency_spans_poll_batches(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """Concurrency must not be confined to a single ``getmany`` batch.

    The binding topic has ONE partition and records arrive one at a time, so a
    fix that only parallelises within a batch would measure green here and
    change nothing on the lane -- every batch would hold one record.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=4,
        batch_size=1,
        declared_concurrency=4,
    )

    assert len(harness.intervals) == 4
    assert count_overlapping_pairs(harness.intervals) > 0, (
        "records delivered one per poll did not overlap; batch-scoped "
        "concurrency is worth nothing on a single-partition topic, which is "
        "exactly what onex.cmd.omnibase-infra.delegation-inference-request.v1 is"
    )


# --------------------------------------------------------------------------
# AC1 safety: the undeclared path must be UNCHANGED, not "concurrency of 1".
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undeclared_path_is_serial(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """A node that declares nothing keeps the pre-OMN-18852 behaviour."""
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=4,
        batch_size=4,
        declared_concurrency=None,
        handler_seconds=0.02,
    )

    assert len(harness.intervals) == 4
    assert count_overlapping_pairs(harness.intervals) == 0, (
        "an undeclared node must execute records serially; this is the "
        "property that makes the change safe to land"
    )
    assert harness.peak_live == 1


@pytest.mark.asyncio
async def test_undeclared_path_spawns_no_dispatch_task(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """Serial must be the inline ``await``, not a semaphore of size one.

    A semaphore of one is a different code path with different cancellation,
    drain and rewind-timing behaviour. The assertion is structural: the loop
    must create no task at all for an undeclared subscription.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=3,
        batch_size=3,
        declared_concurrency=None,
        handler_seconds=0.01,
    )

    assert harness.spawned_tasks == 0, (
        f"the undeclared consume loop spawned {harness.spawned_tasks} task(s). "
        "An undeclared node must take the identical inline await it took "
        "before this change -- not a task spawn under a semaphore of size 1"
    )


@pytest.mark.asyncio
async def test_explicit_bound_of_one_is_also_the_inline_path(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """``max_in_flight_records: 1`` is identical to declaring nothing."""
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=3,
        batch_size=3,
        declared_concurrency=1,
        handler_seconds=0.01,
    )

    assert harness.spawned_tasks == 0
    assert count_overlapping_pairs(harness.intervals) == 0
    assert harness.peak_live == 1


# --------------------------------------------------------------------------
# AC1 rewind safety: fail-closed under concurrency.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewind_under_concurrency_seeks_lowest_and_drains_first(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """A rewind must not race a sibling, and must not seek past a record.

    Offsets 1001 and 1003 both exhaust retries with an unconfirmed DLQ write
    while 1000 and 1002 succeed, all four in flight together. Three things have
    to hold:

    * the seek happens with NO handler still running -- two tasks calling
      ``consumer.seek`` on one TopicPartition concurrently is a race whose
      loser silently wins;
    * the seek targets the LOWEST requesting offset (1001), because seeking to
      1003 would commit past 1001 and lose it -- the OMN-15232 failure this
      mechanism exists to prevent;
    * exactly one seek is issued, not one per requesting record.

    Replaying 1002 and 1003, which already succeeded, is the accepted cost:
    at-least-once is this path's documented delivery contract
    (``_process_consumed_record``), and a duplicate is strictly preferable to
    a lost record.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=4,
        batch_size=4,
        declared_concurrency=4,
        handler_seconds=0.05,
        dlq_unpersisted_offsets=frozenset({BASE_OFFSET + 1, BASE_OFFSET + 3}),
    )

    consumer = harness.consumer
    assert consumer is not None

    assert consumer.seek_calls, (
        "two records exhausted retries with an unconfirmed DLQ write and the "
        "fetch position was allowed to advance past them (OMN-15232)"
    )
    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), BASE_OFFSET + 1)
    ], (
        "the rewind must be a single seek to the LOWEST offset that requested "
        f"one, got {consumer.seek_calls}. Seeking to 1003 would let the "
        "auto-committer commit past 1001, which exists nowhere durable"
    )
    assert consumer.seek_observed_in_flight == [0], (
        "the seek was issued while "
        f"{consumer.seek_observed_in_flight} handler(s) were still running. A "
        "rewind must drain the partition's in-flight work first: concurrent "
        "seek() calls on one TopicPartition race, and a sibling that finishes "
        "after the seek moves the position forward again"
    )


@pytest.mark.asyncio
async def test_rewind_under_concurrency_loses_no_record(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """Every record at or after the rewind point must be redelivered.

    Proven by the seek target rather than by a second fetch: the fixture's
    consumer has no broker behind it, so what is verifiable here is that the
    position was moved to a point from which nothing is skipped.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=5,
        batch_size=5,
        declared_concurrency=5,
        handler_seconds=0.03,
        dlq_unpersisted_offsets=frozenset({BASE_OFFSET + 2}),
    )

    consumer = harness.consumer
    assert consumer is not None
    assert len(consumer.seek_calls) == 1
    _tp, seeked_to = consumer.seek_calls[0]
    assert seeked_to <= BASE_OFFSET + 2, (
        f"seeked to {seeked_to}, past the failed record at {BASE_OFFSET + 2}; "
        "that record exists nowhere durable and is now lost"
    )


@pytest.mark.asyncio
async def test_serial_rewind_behaviour_is_unchanged(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """The undeclared path still rewinds inline and drops the batch remainder.

    This is the OMN-15232 contract, restated as a regression guard: on the
    serial path the seek is issued from inside record processing, and the rest
    of that partition's batch is discarded rather than processed past the
    rewind point.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=4,
        batch_size=4,
        declared_concurrency=None,
        handler_seconds=0.01,
        dlq_unpersisted_offsets=frozenset({BASE_OFFSET + 1}),
    )

    consumer = harness.consumer
    assert consumer is not None
    assert consumer.seek_calls[0] == (
        TopicPartition(TEST_TOPIC, TEST_PARTITION),
        BASE_OFFSET + 1,
    )
    # Records 1002/1003 sit past the rewind point and must not have been
    # dispatched from the stale batch.
    dispatched = [i.offset for i in harness.intervals]
    assert BASE_OFFSET + 2 not in dispatched, (
        "the serial path processed past its own rewind point; the batch "
        "remainder is stale once the fetch position moves back"
    )


# --------------------------------------------------------------------------
# Golden chain and error chain under concurrency.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_golden_chain_every_record_reaches_its_subscriber(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """Concurrency must not drop a record from the chain.

    Eight records, bound of 4, no failures: all eight must be delivered
    exactly once, and the set must be complete -- a dispatch loop that leaks a
    task or loses a batch remainder shows up here as a missing offset rather
    than as an error.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=8,
        batch_size=3,
        declared_concurrency=4,
        handler_seconds=0.02,
    )

    expected = [BASE_OFFSET + i for i in range(8)]
    assert sorted(harness.delivered_offsets) == expected, (
        f"delivered {sorted(harness.delivered_offsets)}, expected {expected}"
    )
    assert len(harness.delivered_offsets) == len(set(harness.delivered_offsets)), (
        "a record was delivered twice with no rewind in play"
    )
    assert count_overlapping_pairs(harness.intervals) > 0


@pytest.mark.asyncio
async def test_error_chain_one_failure_does_not_starve_siblings(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """A failing record must not prevent its concurrent siblings from running.

    The sibling records still complete; the failure is contained to a rewind
    of its own partition. The live analogue is one abandoned local rung: on
    the serial path it held the only slot for 300 s twice in a 25-minute
    window and nothing else ran.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=4,
        batch_size=4,
        declared_concurrency=4,
        handler_seconds=0.05,
        dlq_unpersisted_offsets=frozenset({BASE_OFFSET}),
    )

    assert len(harness.intervals) == 4, (
        "a failing record blocked its siblings; every record in the in-flight "
        "window must still be dispatched"
    )
    assert count_overlapping_pairs(harness.intervals) > 0
    consumer = harness.consumer
    assert consumer is not None
    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), BASE_OFFSET)
    ]


@pytest.mark.asyncio
async def test_shutdown_drains_in_flight_records(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """Shutdown must not abandon a record it already dispatched.

    A task cancelled mid-handler under auto-commit is the drop this whole
    mechanism exists to avoid: the position can advance past work that never
    completed.
    """
    harness = await _drive_consume_loop(
        bus_config,
        mock_producer,
        record_count=4,
        batch_size=4,
        declared_concurrency=4,
        handler_seconds=0.05,
    )

    unfinished = [i.offset for i in harness.intervals if i.finished == float("inf")]
    assert not unfinished, (
        f"the loop returned with offsets {unfinished} still in flight; "
        "shutdown must drain what it dispatched"
    )


# --------------------------------------------------------------------------
# The declaration itself.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declare_refuses_a_bound_below_one(
    bus_config: ModelKafkaEventBusConfig, mock_producer: AsyncMock
) -> None:
    """A bound of 0 admits nothing; that is a wedged consumer, not config."""
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        event_bus = EventBusKafka(config=bus_config)
        with pytest.raises(ValueError, match="max_in_flight_records must be >= 1"):
            event_bus.declare_consume_concurrency(
                topic=TEST_TOPIC, group_id=TEST_GROUP, max_in_flight_records=0
            )


def test_bus_satisfies_the_declarer_protocol(
    bus_config: ModelKafkaEventBusConfig,
) -> None:
    """Auto-wiring narrows the bus structurally; the narrowing must hold."""
    from omnibase_infra.protocols.protocol_consume_concurrency_declarer import (
        ProtocolConsumeConcurrencyDeclarer,
    )

    assert isinstance(
        EventBusKafka(config=bus_config), ProtocolConsumeConcurrencyDeclarer
    )

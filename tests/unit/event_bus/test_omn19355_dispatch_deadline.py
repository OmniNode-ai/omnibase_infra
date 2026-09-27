# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19355: a handler that never returns must not wedge its consumer group.

THE INCIDENT. On the .201 dev lane on 2026-09-23 the ``lab_lane_health``
projection handler parked a ``to_thread`` worker forever (a ``/proc`` read
showed it in ``epoll_wait`` with timeout -1) at offset 51808. The serial consume
loop awaits each handler before it polls again, so it never polled again.
aiokafka evicted the member at ``max_poll_interval_ms`` (1800000) and it never
rejoined, because a rejoin only happens inside the next ``getmany``. Auto-commit
had already committed the fetch position past the batch, so records 51809 to
51832 were never processed: at-most-once loss.

WHAT THESE TESTS HOLD THE LOOP TO.

* A dispatch that outlives its deadline is abandoned, its record is quarantined
  to the DLQ with failure class ``dispatch_deadline_exceeded``, and the NEXT
  record is processed. The loop keeps polling well inside
  ``max_poll_interval_ms``, which is what keeps the member in its group.
* While a serial dispatch is slow, the fetch position is pinned at the
  in-flight record, so auto-commit can never commit past a hung record that has
  no durable quarantine.
* An abandoned dispatch is counted: DEGRADED from the first, UNHEALTHY at the
  declared limit, and back to healthy when it finally returns.

Every behaviour asserted here has a negative control asserting that a normal
handler is unaffected.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.errors import DispatchDeadlineExceededError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

pytestmark = pytest.mark.unit

TOPIC = "onex.evt.omnibase-infra.lab-lane-health.v1"
GROUP = "local.omnimarket.lab_lane_health_projection.consume.1.0.0"
PARTITION = 0
TP = TopicPartition(TOPIC, PARTITION)
#: The live hung coordinate on the .201 dev lane, 2026-09-23.
HUNG = 51808

#: Bound on every drive of the real loop. Before OMN-19355 a hung handler kept
#: the loop inside one dispatch forever, so without this bound the red run of
#: this file would hang instead of failing.
LOOP_BOUND_SECONDS = 15.0


class _PartitionLog:
    """``AIOKafkaConsumer`` stand-in over one partition with a real fetch position.

    ``getmany`` returns up to ``batch_size`` records from the fetch position and
    advances it past them, exactly as a real fetch does, which is the position
    auto-commit would commit. ``seek`` moves it back. Redelivery after a seek is
    therefore modelled by the same mechanism Kafka uses, not by a script.
    """

    def __init__(self, offsets: list[int], *, batch_size: int = 50) -> None:
        self._records = [_raw_msg(offset) for offset in offsets]
        self._batch_size = batch_size
        self.fetch_position = offsets[0] if offsets else 0
        self.seeks: list[int] = []
        self.polled_at: list[float] = []
        self.on_drained: Callable[[], None] | None = None

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[Any]]:
        self.polled_at.append(time.monotonic())
        batch = [r for r in self._records if r.offset >= self.fetch_position][
            : self._batch_size
        ]
        if not batch:
            if self.on_drained is not None:
                self.on_drained()
            return {}
        self.fetch_position = batch[-1].offset + 1
        return {TP: batch}

    def assignment(self) -> set[TopicPartition]:
        return set()

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seeks.append(offset)
        self.fetch_position = offset

    async def stop(self) -> None:
        return None

    def max_poll_gap(self) -> float:
        gaps = [b - a for a, b in zip(self.polled_at, self.polled_at[1:], strict=False)]
        return max(gaps, default=0.0)


def _raw_msg(offset: int) -> MagicMock:
    msg = MagicMock()
    msg.topic = TOPIC
    msg.partition = PARTITION
    msg.offset = offset
    msg.timestamp = int(datetime.now(UTC).timestamp() * 1000)
    msg.key = None
    msg.headers = []
    msg.value = b'{"correlation_id": "fc3d267f-dcb2-44c8-ac6f-9285a9b5e827"}'
    return msg


def _config(**overrides: Any) -> ModelKafkaEventBusConfig:
    values: dict[str, Any] = {
        "bootstrap_servers": "localhost:9092",
        # The smallest poll interval the config admits, so a wedge that would
        # outlive it is caught by the gap assertion rather than by patience.
        "max_poll_interval_ms": 10_000,
        "session_timeout_ms": 6_000,
        "heartbeat_interval_ms": 2_000,
        "consumer_dispatch_deadline_seconds": 0.4,
        "consumer_dispatch_withhold_after_seconds": 0.05,
    }
    values.update(overrides)
    return ModelKafkaEventBusConfig(**values)


class _Harness:
    """The real ``EventBusKafka`` consume loop over a ``_PartitionLog``."""

    def __init__(
        self,
        config: ModelKafkaEventBusConfig,
        log: _PartitionLog,
        handler: Callable[[int], Any],
        *,
        dlq_results: list[bool] | None = None,
    ) -> None:
        self.config = config
        self.log = log
        self.handler = handler
        self.dlq_calls: list[dict[str, Any]] = []
        self._dlq_results = list(dlq_results or [])
        self.bus: EventBusKafka | None = None

    async def run(self, *, concurrency: int = 1) -> None:
        producer = AsyncMock()
        # An open producer, so ``healthy`` measures the deadline and not the mock.
        producer._closed = False
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=producer,
        ):
            bus = EventBusKafka(config=self.config)
            self.bus = bus
            await bus.start()
            bus._group_consumers[(TOPIC, GROUP)] = self.log  # type: ignore[assignment]
            if concurrency > 1:
                bus.declare_consume_concurrency(
                    topic=TOPIC, group_id=GROUP, max_in_flight_records=concurrency
                )
            self.log.on_drained = lambda: setattr(bus, "_shutdown", True)

            async def _callback(message: Any) -> None:
                await self.handler(message.raw_offset)

            bus._subscribers[TOPIC] = [(GROUP, "sub-1", _callback)]  # type: ignore[assignment]

            def _to_model(msg: Any, _topic: str) -> MagicMock:
                message = MagicMock()
                message.raw_offset = msg.offset
                message.headers.retry_count = 0
                message.headers.max_retries = 3
                return message

            async def _record_dlq(**kwargs: Any) -> bool:
                self.dlq_calls.append(kwargs)
                return self._dlq_results.pop(0) if self._dlq_results else True

            with (
                patch.object(bus, "_kafka_msg_to_model", side_effect=_to_model),
                patch.object(bus, "_publish_to_dlq", side_effect=_record_dlq),
                patch.object(
                    event_bus_kafka_module,
                    "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS",
                    0.0,
                ),
            ):
                await asyncio.wait_for(
                    bus._consume_loop(TOPIC, GROUP, uuid4()),
                    timeout=LOOP_BOUND_SECONDS,
                )

    async def close(self) -> None:
        if self.bus is not None:
            await self.bus.close()


class _Handlers:
    """Handlers keyed by offset: some hang in a worker thread, the rest return."""

    def __init__(self, *, hung: set[int], slow: dict[int, float] | None = None):
        self.hung = hung
        self.slow = slow or {}
        self.handled: list[int] = []
        self.entered: list[int] = []
        # The live shape: the handler is parked in a to_thread worker, which is
        # the one thing the consume loop cannot cancel.
        self.release = threading.Event()

    async def __call__(self, offset: int) -> None:
        self.entered.append(offset)
        if offset in self.hung:
            await asyncio.to_thread(self.release.wait)
            return
        if offset in self.slow:
            await asyncio.sleep(self.slow[offset])
        self.handled.append(offset)


async def _settle(bus: EventBusKafka) -> None:
    """Let abandoned dispatches that were just released finish."""
    for _ in range(100):
        if bus.dispatch_deadline_status().orphaned_dispatches == 0:
            return
        await asyncio.sleep(0.02)


# --- the deadline ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hung_handler_is_quarantined_and_the_next_record_is_processed() -> None:
    """The live shape: 51808 hangs in a worker thread, 51809 and 51810 follow it."""
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1, HUNG + 2])
    harness = _Harness(_config(), log, handlers)
    try:
        await harness.run()
        assert handlers.handled == [HUNG + 1, HUNG + 2], (
            "the records after the hung one were never processed -- the 51809 "
            "to 51832 loss of 2026-09-23"
        )
        assert len(harness.dlq_calls) == 1
        quarantine = harness.dlq_calls[0]
        assert (
            quarantine["failure_class"]
            == EnumDlqFailureClass.DISPATCH_DEADLINE_EXCEEDED
        )
        assert isinstance(quarantine["error"], DispatchDeadlineExceededError)
        assert str(HUNG) in quarantine["validation_detail"], (
            "the quarantine must name the record it abandoned"
        )
        assert log.max_poll_gap() < harness.config.max_poll_interval_ms / 1000.0, (
            "the loop went longer than max_poll_interval_ms without polling, so "
            "a real broker would have evicted the member"
        )
        assert harness.bus is not None
        status = harness.bus.dispatch_deadline_status()
        assert status.orphaned_dispatches == 1
        assert status.deadline_expiries_total == 1
        assert status.status == "degraded"
    finally:
        handlers.release.set()
        await harness.close()


@pytest.mark.asyncio
async def test_a_hung_record_is_never_committed_past_while_it_runs() -> None:
    """The fetch position auto-commit reads is pinned at the in-flight record.

    After ``getmany`` the fetch position is past the whole batch, and that is
    what auto-commit commits. Once the dispatch outlives the withhold threshold
    the loop seeks it back, so a crash during the hang redelivers the record
    instead of losing it and everything fetched with it.
    """
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1, HUNG + 2])
    config = _config()
    harness = _Harness(config, log, handlers)
    sampled: list[int] = []

    async def _sample_during_hang() -> None:
        while HUNG not in handlers.entered:
            await asyncio.sleep(0.005)
        await asyncio.sleep(config.consumer_dispatch_withhold_after_seconds + 0.1)
        sampled.append(log.fetch_position)

    sampler = asyncio.create_task(_sample_during_hang())
    try:
        await harness.run()
        await sampler
        assert sampled == [HUNG], (
            f"while 51808 was hung auto-commit would have committed "
            f"{sampled}; only the hung record's own offset is safe"
        )
        assert log.seeks[0] == HUNG
        assert HUNG + 1 in log.seeks, (
            "after the confirmed quarantine the position must move past the "
            "record, or it is redelivered and abandoned again"
        )
    finally:
        handlers.release.set()
        await harness.close()


@pytest.mark.asyncio
async def test_an_unconfirmed_quarantine_rewinds_instead_of_advancing() -> None:
    """OMN-15232 is not relaxed: a record with no durable copy is redelivered."""
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1])
    harness = _Harness(_config(), log, handlers, dlq_results=[False, True])
    try:
        await harness.run()
        assert len(harness.dlq_calls) == 2, (
            "the first quarantine was not confirmed, so the record must be "
            "redelivered and quarantined again"
        )
        assert handlers.entered.count(HUNG) == 2
        assert handlers.handled == [HUNG + 1]
        assert harness.bus is not None
        assert harness.bus.dispatch_deadline_status().orphaned_dispatches == 2
    finally:
        handlers.release.set()
        await harness.close()


# --- negative controls -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_normal_handler_is_unaffected() -> None:
    """Negative control: fast handlers see one fetch, no seek, no quarantine."""
    handlers = _Handlers(hung=set())
    log = _PartitionLog([HUNG, HUNG + 1, HUNG + 2])
    harness = _Harness(_config(), log, handlers)
    try:
        await harness.run()
        assert handlers.handled == [HUNG, HUNG + 1, HUNG + 2]
        assert harness.dlq_calls == []
        assert log.seeks == [], "a fast dispatch must not touch the fetch position"
        assert len(log.polled_at) == 2, "one batch and one empty poll, no refetch"
        assert harness.bus is not None
        status = harness.bus.dispatch_deadline_status()
        assert status.orphaned_dispatches == 0
        assert status.deadline_expiries_total == 0
        assert status.status == "healthy"
        health = await harness.bus.health_check()
        assert health["degraded"] is False
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_a_slow_handler_inside_its_deadline_is_processed_exactly_once() -> None:
    """Negative control: slow but finishing is not hung.

    It outlives the withhold threshold, so the position is pinned and the batch
    refetched from the next record, but it is neither quarantined nor
    redelivered, and nothing is counted against health.
    """
    handlers = _Handlers(hung=set(), slow={HUNG: 0.15})
    log = _PartitionLog([HUNG, HUNG + 1, HUNG + 2])
    harness = _Harness(_config(), log, handlers)
    try:
        await harness.run()
        assert handlers.handled == [HUNG, HUNG + 1, HUNG + 2]
        assert harness.dlq_calls == []
        assert log.seeks == [HUNG, HUNG + 1]
        assert harness.bus is not None
        assert harness.bus.dispatch_deadline_status().deadline_expiries_total == 0
    finally:
        await harness.close()


# --- health ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orphans_degrade_then_reach_unhealthy_then_recover() -> None:
    """A Python thread cannot be killed, so the count is bounded by health.

    DEGRADED from the first abandoned dispatch; UNHEALTHY at the limit so the
    supervisor replaces the process before parked threads exhaust the projection
    gate; healthy again once they return on their own.
    """
    handlers = _Handlers(hung={HUNG, HUNG + 1})
    log = _PartitionLog([HUNG, HUNG + 1, HUNG + 2])
    harness = _Harness(_config(consumer_dispatch_orphan_limit=2), log, handlers)
    try:
        await harness.run()
        assert handlers.handled == [HUNG + 2]
        assert harness.bus is not None
        status = harness.bus.dispatch_deadline_status()
        assert status.orphaned_dispatches == 2
        assert status.status == "unhealthy"
        assert len(status.orphans) == 2
        assert str(HUNG) in status.orphans[0]
        health = await harness.bus.health_check()
        assert health["healthy"] is False
        assert health["dispatch_deadline"]["orphaned_dispatches"] == 2  # type: ignore[index]

        handlers.release.set()
        await _settle(harness.bus)
        status = harness.bus.dispatch_deadline_status()
        assert status.orphaned_dispatches == 0
        assert status.deadline_expiries_total == 2
        assert status.status == "healthy"
    finally:
        handlers.release.set()
        await harness.close()


@pytest.mark.asyncio
async def test_one_orphan_reports_degraded_not_unhealthy() -> None:
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1])
    harness = _Harness(_config(), log, handlers)
    try:
        await harness.run()
        assert harness.bus is not None
        health = await harness.bus.health_check()
        assert health["healthy"] is True
        assert health["degraded"] is True
    finally:
        handlers.release.set()
        await harness.close()


# --- the poll budget -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_long_batch_is_cut_and_refetched_before_the_poll_budget() -> None:
    """No run of records can carry the gap between polls past the eviction.

    The loop does not start a record unless its full deadline still fits inside
    the batch budget. Here each record is fast but the deadline is most of the
    poll interval, so the batch is cut after about a second and the rest
    refetched -- each record exactly once.
    """
    offsets = list(range(HUNG, HUNG + 40))
    handlers = _Handlers(hung=set(), slow=dict.fromkeys(offsets, 0.04))
    log = _PartitionLog(offsets)
    config = _config(
        consumer_dispatch_deadline_seconds=8.2,
        consumer_dispatch_withhold_after_seconds=5.0,
    )
    harness = _Harness(config, log, handlers)
    try:
        await harness.run()
        assert handlers.handled == offsets, "every record exactly once, in order"
        assert harness.dlq_calls == []
        assert len(log.seeks) >= 1, "the batch was never cut"
        assert len(log.polled_at) >= 3
    finally:
        await harness.close()


# --- the concurrent path -----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_concurrent_path_quarantines_a_hung_record_and_drains() -> None:
    """OMN-18852's driver drains every task before it exits or seeks.

    A task that never returned made that drain wait forever. With the deadline
    the task ends at its deadline and the driver finishes.
    """
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1, HUNG + 2])
    harness = _Harness(_config(), log, handlers)
    try:
        await harness.run(concurrency=2)
        assert sorted(handlers.handled) == [HUNG + 1, HUNG + 2]
        assert len(harness.dlq_calls) == 1
        assert (
            harness.dlq_calls[0]["failure_class"]
            == EnumDlqFailureClass.DISPATCH_DEADLINE_EXCEEDED
        )
        assert harness.bus is not None
        assert harness.bus.dispatch_deadline_status().orphaned_dispatches == 1
    finally:
        handlers.release.set()
        await harness.close()


# --- config ----------------------------------------------------------------------


def test_the_deadline_never_reaches_the_eviction() -> None:
    """600s on the 1800s lanes; capped under the 300s library default."""
    lane = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092", max_poll_interval_ms=1_800_000
    )
    assert lane.effective_dispatch_deadline_seconds == 600.0
    library_default = ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    assert library_default.max_poll_interval_ms == 300_000
    assert library_default.effective_dispatch_deadline_seconds == pytest.approx(255.0)
    assert library_default.effective_dispatch_deadline_seconds > 240.0, (
        "node_delegate_skill_orchestrator bounds itself to 240s; a deadline "
        "below that would quarantine legitimate delegations"
    )

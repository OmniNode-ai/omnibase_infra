# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-16422 -- HandlerDlqReplay.run() is a whole-topic batch drain auto-wired
as a PER-MESSAGE trigger on the DLQ topic. On a self-feeding DLQ (messages
that get replayed, fail again downstream, and land right back on the same DLQ
topic) the topic never goes idle, so the pre-fix loop never returns and never
commits -- starving the outer trigger consumer's heartbeat while every new
DLQ arrival piles another unbounded drain attempt on top (root cause of the
OMN-16418 poison-pill storm).

These tests reproduce the storm precondition -- a DLQ topic that never goes
idle -- with a fake consumer whose ``consume_messages()`` never terminates on
its own, and assert the fix (bounded record count + bounded wall-clock +
incremental commit) makes a single ``run()`` invocation always return
quickly, always commit progress, and never balloon into an unbounded re-drain
regardless of how many times it is re-triggered.

Ticket: OMN-16422
Evidence-Ticket: OMN-16422
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.unit


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": "onex.dlq.omnibase-infra.events.v1",
        "max_replay_count": 5,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _message(topic: str = "dev.orders.command.v1") -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=topic,
        original_key="k",
        original_value='{"hello": "world"}',
        original_offset="10",
        original_partition=0,
        failure_reason="boom",
        failure_timestamp="2026-06-02T00:00:00Z",
        correlation_id=uuid4(),
        retry_count=0,
        error_type="InfraConnectionError",  # retryable -> always replay-eligible
        dlq_offset=42,
        dlq_partition=1,
        raw_payload={"original_topic": topic},
    )


class _NeverIdleConsumer:
    """Simulates a DLQ topic that NEVER goes quiet -- the exact live storm
    precondition. Yields a fresh distinct message forever; pre-fix, the
    ``async for`` loop in ``run()`` would never terminate on its own against
    a consumer shaped like this (no ``limit`` was applied by default).
    """

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._started = False
        self.commits = 0
        self.yielded = 0

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        while True:
            self.yielded += 1
            yield _message()

    async def commit(self) -> None:
        self.commits += 1


class _SlowNeverIdleConsumer(_NeverIdleConsumer):
    """Same never-idle shape, but each yield is separated by a real sleep so
    the wall-clock bound trips before the record-count bound would.
    """

    def __init__(self, config: ModelDlqReplayEngineConfig, *, delay: float) -> None:
        super().__init__(config)
        self._delay = delay

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        while True:
            await asyncio.sleep(self._delay)
            self.yielded += 1
            yield _message()


class _NoopEffect:
    """Fake producer / quarantine-producer: succeeds without publishing
    anything real. Every synthetic message in this module is replay-eligible
    (retryable error type, retry_count below max), so only ``replay_message``
    is exercised.
    """

    def __init__(self) -> None:
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def replay_message(
        self, message: object, replay_correlation_id: object
    ) -> None:
        return None

    async def quarantine_message(
        self, message: object, reason: str, quarantine_correlation_id: object
    ) -> None:
        return None


def _handler(consumer: object, config: ModelDlqReplayEngineConfig) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumer=consumer,  # type: ignore[arg-type]
        producer=_NoopEffect(),  # type: ignore[arg-type]
        quarantine_producer=_NoopEffect(),  # type: ignore[arg-type]
        tracking=None,
    )


async def test_never_idle_topic_still_returns_a_bounded_batch() -> None:
    """The storm precondition: a topic that never goes idle. Pre-fix this
    would hang forever (no default ``limit``). Post-fix, run() returns after
    exactly ``max_records_per_run`` records -- proving the invocation is
    bounded regardless of how busy the topic is.
    """
    config = _config(max_records_per_run=10, commit_every_n_records=25)
    consumer = _NeverIdleConsumer(config)
    handler = _handler(consumer, config)

    result = await asyncio.wait_for(handler.run(), timeout=5.0)

    assert result.total_processed == 10
    # OMN-17137: exactly the bound, with no discarded prefetch. The original
    # ``async for`` shape pulled one extra record from the generator before
    # the body re-checked the bound and broke (yielded == 11). Bounding the
    # WAIT means both bounds are now checked BEFORE the next record is
    # requested, so the eleventh record is never pulled and re-delivered.
    assert consumer.yielded == 10


async def test_never_idle_topic_commits_incrementally_not_only_at_the_end() -> None:
    """AC: the persistent consumer group's committed offset must actually
    advance under continuous self-feeding, not stay perpetually uncommitted.
    Proven here by observing MULTIPLE commit() calls within one bounded run,
    not just one commit after the (never-reached, pre-fix) idle point.
    """
    config = _config(max_records_per_run=23, commit_every_n_records=5)
    consumer = _NeverIdleConsumer(config)
    handler = _handler(consumer, config)

    result = await asyncio.wait_for(handler.run(), timeout=5.0)

    assert result.total_processed == 23
    # 4 incremental commits (at 5, 10, 15, 20) + 1 final commit for the
    # trailing 3 uncommitted records = 5 total. The key assertion is ">1":
    # progress is committed DURING the run, not only after it ends.
    assert consumer.commits == 5


async def test_bounded_run_is_re_entrant_without_compounding() -> None:
    """Simulates repeated per-message triggers against the SAME never-idle
    topic (each DLQ arrival independently invoking run() again, as the live
    auto-wiring boundary does). Each invocation must stay independently
    bounded -- proving the fix does NOT cause unbounded re-drain / re-trigger
    even when triggered back-to-back many times.
    """
    config = _config(max_records_per_run=8, commit_every_n_records=4)
    consumer = _NeverIdleConsumer(config)
    handler = _handler(consumer, config)

    for _ in range(5):
        result = await asyncio.wait_for(handler.run(), timeout=5.0)
        assert result.total_processed == 8

    # Five independently-bounded invocations processed exactly 5*8 = 40
    # records -- not an ever-growing/unbounded cumulative drain. Exactly 40
    # (not 45) since OMN-17137 removed the per-run discarded prefetch.
    assert consumer.yielded == 40


async def test_wall_clock_bound_trips_before_record_count_bound() -> None:
    """A topic that yields messages slowly (but never idles) must be cut off
    by the wall-clock bound even when the record-count bound is far from
    reached -- proving run() cannot be starved indefinitely by a topic that
    trickles just fast enough to keep yielding, but slow enough that draining
    to max_records_per_run would take too long.

    (The pre-OMN-17137 wording here said "never hit consumer_timeout_ms
    idle". That premise was false: in aiokafka ``consumer_timeout_ms`` is the
    background fetcher's max wait, not an idle-iteration timeout, so there is
    no idle cutoff for this fake to stay ahead of. See
    test_handler_dlq_replay_idle_wait_omn17137.py.)
    """
    config = _config(
        max_records_per_run=1000,
        max_run_duration_seconds=0.2,
        commit_every_n_records=1000,
    )
    consumer = _SlowNeverIdleConsumer(config, delay=0.05)
    handler = _handler(consumer, config)

    started = time.monotonic()
    result = await asyncio.wait_for(handler.run(), timeout=5.0)
    elapsed = time.monotonic() - started

    assert result.total_processed < 1000
    assert result.total_processed >= 1
    # Bounded well under what an unbounded 1000-record drain at 0.05s/record
    # (50s) would take -- proves the wall-clock bound, not the count bound,
    # ended this run.
    assert elapsed < 2.0


async def test_explicit_limit_still_narrows_below_the_default_bound() -> None:
    """An explicit ``limit`` (e.g. a deliberate small manual run) still wins
    when it is stricter than ``max_records_per_run`` -- backward-compatible
    with the pre-fix ``limit`` semantics.
    """
    config = _config(limit=3, max_records_per_run=200, commit_every_n_records=25)
    consumer = _NeverIdleConsumer(config)
    handler = _handler(consumer, config)

    result = await asyncio.wait_for(handler.run(), timeout=5.0)

    assert result.total_processed == 3


async def test_dry_run_never_commits_even_when_bounded() -> None:
    """dry_run must still publish/commit nothing -- the bound changes when
    work stops, not whether dry_run's no-mutation invariant holds.
    """
    config = _config(max_records_per_run=6, commit_every_n_records=2, dry_run=True)
    consumer = _NeverIdleConsumer(config)
    handler = _handler(consumer, config)

    result = await asyncio.wait_for(handler.run(), timeout=5.0)

    assert result.total_processed == 6
    assert result.dry_run is True
    assert consumer.commits == 0

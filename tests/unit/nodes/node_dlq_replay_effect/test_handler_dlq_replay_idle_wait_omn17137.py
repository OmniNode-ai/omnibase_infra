# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-17137 -- the OMN-16422 wall-clock bound was unenforceable on an IDLE
DLQ topic.

OMN-16422 bounded ``HandlerDlqReplay.run()`` by record count and wall clock,
but checked the wall-clock deadline only INSIDE the ``async for`` body -- i.e.
only after a record was yielded. ``DLQConsumer.consume_messages()`` iterates an
``AIOKafkaConsumer`` whose ``__anext__`` is ``while True: return await
self.getone()``: it blocks indefinitely until a record arrives. The
``consumer_timeout_ms=5000`` passed at construction does NOT end that
iteration -- in aiokafka that parameter is the background fetching routine's
max wait (default 200ms), not kafka-python's idle-iteration timeout.

So the moment the topic went idle mid-drain, ``run()`` parked in ``getone()``
forever and the deadline was never evaluated again. Because the auto-wired
outer trigger consumer's ``_consume_loop`` is a serial ``async for msg in
consumer:`` that awaits each dispatch, a parked ``run()`` held the entire loop
until aiokafka evicted it at ``max_poll_interval_ms``.

Live evidence (stability lane, 2026-08-30, ``KAFKA_MAX_POLL_INTERVAL_MS=1800000``)::

    08:13:10  Joined group ...__t.onex.dlq.omnibase-infra.events.v1 (generation 338)
    08:13:10  Starting DLQ consumer for topic: onex.dlq.omnibase-infra.events.v1
    08:13:10  QUARANTINED d0235763-... (Exceeded max replay count: 5 >= 5)
    08:13:16  QUARANTINED 0e980da0-... (Exceeded max replay count: 5 >= 5)
    ...       30 minutes of silence -- the 10.0s wall-clock bound never fires
    08:43:10  OffsetCommit failed ... UnknownMemberIdError ... will rejoin
    08:43:10  Revoking previously assigned partitions  ->  group state Empty

``rpk group describe`` then reported that group ``Empty`` / 0 members with
lag 403, which the runtime health monitor surfaces as BOTH
``empty_consumer_groups`` and ``topic_coverage`` DEGRADED -- the container
healthcheck exits 1 and the lane goes ``unhealthy``.

The existing OMN-16422 suite cannot catch this: every fake there yields
forever (the never-idle shape), so the loop body always runs and always
re-checks the deadline. The fake below reproduces the shape that actually
broke -- a consumer that yields a few records and then BLOCKS WITHOUT
YIELDING.

Ticket: OMN-17137
Evidence-Ticket: OMN-17137
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

# Every assertion below must fail LOUDLY rather than hang the suite forever if
# the bound regresses, so each run() is wrapped in an outer wait_for whose
# timeout is far above the run's own budget but far below "forever".
_OUTER_TIMEOUT_SECONDS = 5.0


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": "onex.dlq.omnibase-infra.events.v1",
        "max_replay_count": 5,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _message() -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic="dev.orders.command.v1",
        original_key="k",
        original_value='{"hello": "world"}',
        original_offset="10",
        original_partition=0,
        failure_reason="boom",
        failure_timestamp="2026-06-02T00:00:00Z",
        correlation_id=uuid4(),
        retry_count=0,
        error_type="InfraConnectionError",  # retryable -> replay-eligible
        dlq_offset=42,
        dlq_partition=1,
        raw_payload={"original_topic": "dev.orders.command.v1"},
    )


class _GoesIdleConsumer:
    """The shape that actually broke: yields ``burst`` records, then BLOCKS.

    This is aiokafka's real behavior on an idle topic -- ``__anext__`` awaits
    ``getone()``, which never returns until a record arrives. Nothing here
    raises, nothing times out on its own, and nothing yields again. A run()
    that only re-checks its deadline between yielded records can never escape.
    """

    def __init__(self, config: ModelDlqReplayEngineConfig, *, burst: int) -> None:
        self.config = config
        self._burst = burst
        self._started = False
        self.commits = 0
        self.yielded = 0
        self.parked = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        for _ in range(self._burst):
            self.yielded += 1
            yield _message()
        # Idle forever, exactly like a quiet Kafka partition.
        self.parked = True
        await asyncio.Event().wait()

    async def commit(self) -> None:
        self.commits += 1


class _ImmediatelyIdleConsumer(_GoesIdleConsumer):
    """Idle from the first record on -- the topic was already quiet when the
    per-message trigger fired (the drain has nothing left to do).
    """

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        super().__init__(config, burst=0)


class _NoopEffect:
    """Fake producer / quarantine-producer: succeeds without publishing."""

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


def _handler(consumer: object) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumer=consumer,  # type: ignore[arg-type]
        producer=_NoopEffect(),  # type: ignore[arg-type]
        quarantine_producer=_NoopEffect(),  # type: ignore[arg-type]
        tracking=None,
    )


async def test_run_returns_when_the_topic_goes_idle_mid_drain() -> None:
    """THE regression test. Pre-fix this hangs until the outer wait_for kills
    it; post-fix run() returns at its own wall-clock bound.

    The record-count bound is set far above the burst so it can never be what
    ends the run -- only the wall-clock bound applied to the WAIT can.
    """
    config = _config(
        max_records_per_run=1000,
        max_run_duration_seconds=0.3,
        commit_every_n_records=1000,
    )
    consumer = _GoesIdleConsumer(config, burst=3)
    handler = _handler(consumer)

    started = time.monotonic()
    result = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started

    # The burst was drained, then the run ended on its own budget.
    assert result.total_processed == 3
    assert consumer.parked is True, (
        "the fake must have reached its idle park -- otherwise this test is "
        "exercising the never-idle shape OMN-16422 already covered"
    )
    # Ended at the wall-clock bound, not by hanging: comfortably under the
    # outer guard, and at least the budget it was given.
    assert elapsed >= 0.3
    assert elapsed < _OUTER_TIMEOUT_SECONDS / 2


async def test_idle_run_still_commits_the_records_it_did_process() -> None:
    """Ending on the idle bound must not throw away committed progress: the
    trailing uncommitted records are committed before run() returns, so the
    persistent group's offset advances instead of replaying them forever.
    """
    config = _config(
        max_records_per_run=1000,
        max_run_duration_seconds=0.3,
        commit_every_n_records=1000,  # above the burst: only the final commit fires
    )
    consumer = _GoesIdleConsumer(config, burst=4)
    handler = _handler(consumer)

    result = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)

    assert result.total_processed == 4
    assert consumer.commits == 1


async def test_run_returns_when_the_topic_is_idle_from_the_start() -> None:
    """A trigger that fires against an already-quiet topic must return an
    empty bounded batch, not park. This is the steady-state case once the
    backlog is drained -- the one that held the live consumer for 30 minutes.
    """
    config = _config(max_records_per_run=200, max_run_duration_seconds=0.2)
    consumer = _ImmediatelyIdleConsumer(config)
    handler = _handler(consumer)

    result = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)

    assert result.total_processed == 0
    assert consumer.commits == 0


async def test_repeated_idle_triggers_stay_independently_bounded() -> None:
    """The live boundary re-invokes run() on every DLQ arrival. Each idle
    invocation must return on its own budget without compounding -- five
    back-to-back runs against an always-idle topic must cost ~5 budgets, not
    wedge the outer consumer.
    """
    config = _config(max_records_per_run=200, max_run_duration_seconds=0.1)
    consumer = _ImmediatelyIdleConsumer(config)
    handler = _handler(consumer)

    started = time.monotonic()
    for _ in range(5):
        result = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)
        assert result.total_processed == 0
    elapsed = time.monotonic() - started

    assert elapsed < _OUTER_TIMEOUT_SECONDS / 2


async def test_aiokafka_consumer_timeout_ms_does_not_end_iteration() -> None:
    """Pin the false premise this bug rested on.

    ``DLQConsumer`` constructs its ``AIOKafkaConsumer`` with
    ``consumer_timeout_ms=5000``, and the pre-fix ``run()`` docstring claimed
    the drain "looped until the topic went quiet for ``consumer_timeout_ms``
    (5s)". That is kafka-python's semantics. In aiokafka the parameter is the
    background fetching routine's max wait and ``__anext__`` blocks forever,
    so no idle cutoff exists at any layer below ``run()``. If a future
    aiokafka ever grows one, this test fails and the belt-and-braces timeout
    in ``run()`` can be revisited deliberately rather than by accident.
    """
    import inspect

    from aiokafka import AIOKafkaConsumer

    source = inspect.getsource(AIOKafkaConsumer.__anext__)
    assert "while True" in source
    assert "await self.getone()" in source
    # No timeout/deadline machinery in the iteration path.
    assert "timeout" not in source

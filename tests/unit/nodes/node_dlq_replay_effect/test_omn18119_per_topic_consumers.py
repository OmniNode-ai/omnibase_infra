# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18119 -- three declared subscribe topics, one consumer, one topic drained.

``contract.yaml`` declares three DLQ subscribe topics and, since OMN-18013,
three per-topic dispatcher entries scoped by ``topic:``. ``service_kernel``
keys the materialized dependency map by handler NAME, so all three entries
resolved to a single ``DLQConsumer``, and ``HandlerDlqReplay`` took ``consumer``
singular and read ``self._config = consumer.config``. Whichever entry fired,
the drain subscribed to the events topic and nothing else.

Live on the .201 dev lane at 2026-09-10T01:40Z the persistent replay group held
a committed offset for exactly ONE topic-partition, while the commands DLQ held
1,661 records and had never had an offset committed against it at all. Nothing
was failing. Nothing was logged. Two thirds of the declared surface was simply
not being consumed.

The three bounds below are the whole design, and each exists because the naive
version of this fix breaks something that currently works:

  * a SHARED wall clock, because multiplying the run budget by the topic count
    is how OMN-17137 comes back;
  * a PER-TOPIC record budget, because a shared one is spent entirely by the
    first topic in the order and on this lane that topic is a 700,000-record
    backlog;
  * a short idle probe on a topic's FIRST record, because two of the three
    declared topics are usually empty and an empty topic that waits on the full
    remaining budget starves the topic behind it.

And the order ROTATES, because a topic whose predecessor can always fill its own
record budget is never reached at all. That is not hypothetical: it is the
steady state of this lane while the backlog drains.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQ_REPLAY_CONSUMER_GROUP,
    DLQConsumer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)

pytestmark = pytest.mark.unit

_ORIGINAL_TOPIC = "onex.evt.omniclaude.tool-executed.v1"  # onex-topic-allow: quotes the measured live record
_EVENTS_DLQ = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: the live DLQ this node drains
_INTENTS_DLQ = "onex.dlq.omnibase-infra.intents.v1"  # onex-topic-allow: declared and, before this ticket, never drained
_COMMANDS_DLQ = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: declared, 1661 records retained, never drained


def _config(topic: str, **overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": topic,
        "consumer_group": DLQ_REPLAY_CONSUMER_GROUP,
        "max_records_per_run": 3,
        "commit_every_n_records": 1000,
        "max_run_duration_seconds": 5.0,
        "idle_probe_seconds": 0.05,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _payload(*, retry_count: int = 99) -> bytes:
    """Ineligible by retry count, so it quarantines without a replay publish."""
    return json.dumps(
        {
            "original_topic": _ORIGINAL_TOPIC,
            "original_message": {"key": "k", "value": '{"hello": "world"}'},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": retry_count,
        }
    ).encode("utf-8")


class _FakeConsumerRecord:
    def __init__(self, value: bytes, offset: int, partition: int = 0) -> None:
        self.value = value
        self.offset = offset
        self.partition = partition


class _FakeAIOKafkaConsumer:
    """``block=True`` never yields, reproducing a declared-but-empty topic."""

    def __init__(
        self, records: list[_FakeConsumerRecord], *, block: bool = False
    ) -> None:
        self._records = records
        self._index = 0
        self._block = block
        self.position = 0
        self.commits: list[Any] = []

    def __aiter__(self) -> _FakeAIOKafkaConsumer:
        return self

    async def __anext__(self) -> _FakeConsumerRecord:
        if self._block:
            await asyncio.Event().wait()
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        self.position = record.offset + 1
        return record

    async def commit(self, offsets: Any = None) -> None:
        self.commits.append(self.position if offsets is None else offsets)

    async def stop(self) -> None:  # pragma: no cover - not reached
        return None


def _consumer(
    topic: str, offsets: list[int], *, block: bool = False
) -> tuple[DLQConsumer, _FakeAIOKafkaConsumer]:
    consumer = DLQConsumer(_config(topic))
    fake = _FakeAIOKafkaConsumer(
        [_FakeConsumerRecord(_payload(), offset) for offset in offsets], block=block
    )
    consumer._consumer = fake  # type: ignore[assignment]
    consumer._started = True
    return consumer, fake


class _NoopReplayProducer:
    def __init__(self) -> None:
        self._started = True

    async def start(self) -> None:  # pragma: no cover - not reached
        return None

    async def stop(self) -> None:  # pragma: no cover - not reached
        return None

    async def replay_message(
        self, message: object, _cid: object
    ) -> None:  # pragma: no cover
        return None


class _RecordingQuarantineProducer:
    def __init__(self) -> None:
        self._started = True
        self.reasons: list[str] = []

    async def start(self) -> None:  # pragma: no cover - not reached
        return None

    async def stop(self) -> None:  # pragma: no cover - not reached
        return None

    async def quarantine_message(
        self, message: object, reason: str, _cid: object
    ) -> object:
        self.reasons.append(reason)
        return object()

    async def quarantine_unparseable_record(
        self, record: object, _cid: object
    ) -> object:  # pragma: no cover - not reached here
        return object()


def _handler(consumers: dict[str, DLQConsumer]) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumers=consumers,
        producer=_NoopReplayProducer(),  # type: ignore[arg-type]
        quarantine_producer=_RecordingQuarantineProducer(),  # type: ignore[arg-type]
        tracking=None,
    )


def _committed(fake: _FakeAIOKafkaConsumer) -> dict[tuple[str, int], int]:
    merged: dict[tuple[str, int], int] = {}
    for commit in fake.commits:
        assert isinstance(commit, dict), commit
        merged.update(commit)
    return merged


# --------------------------------------------------------------------------
# AC2 -- one run drains EVERY declared topic, and commits per topic.
# RED on origin/dev: HandlerDlqReplay takes no ``consumers`` argument.
# --------------------------------------------------------------------------


async def test_one_run_drains_every_declared_topic() -> None:
    events, events_fake = _consumer(_EVENTS_DLQ, [100, 101])
    intents, intents_fake = _consumer(_INTENTS_DLQ, [10])
    commands, commands_fake = _consumer(_COMMANDS_DLQ, [7, 8])

    result = await _handler(
        {_EVENTS_DLQ: events, _INTENTS_DLQ: intents, _COMMANDS_DLQ: commands}
    ).run()

    assert result.total_processed == 5, result
    assert result.quarantined == 5, result
    assert set(result.topics_drained) == {_EVENTS_DLQ, _INTENTS_DLQ, _COMMANDS_DLQ}
    assert result.dlq_topic == result.topics_drained[0]

    assert _committed(events_fake) == {(_EVENTS_DLQ, 0): 102}, events_fake.commits
    assert _committed(intents_fake) == {(_INTENTS_DLQ, 0): 11}, intents_fake.commits
    assert _committed(commands_fake) == {(_COMMANDS_DLQ, 0): 9}, commands_fake.commits


async def test_each_commit_is_keyed_by_the_topic_the_record_came_from() -> None:
    """A commit keyed by a handler-wide 'primary' topic would ack offsets on a
    topic the record was never read from. Each consumer sees only its own key."""
    events, events_fake = _consumer(_EVENTS_DLQ, [500])
    commands, commands_fake = _consumer(_COMMANDS_DLQ, [3])

    await _handler({_EVENTS_DLQ: events, _COMMANDS_DLQ: commands}).run()

    assert set(_committed(events_fake)) == {(_EVENTS_DLQ, 0)}
    assert set(_committed(commands_fake)) == {(_COMMANDS_DLQ, 0)}


# --------------------------------------------------------------------------
# AC3 -- a declared-but-empty topic costs a probe, not the run.
# --------------------------------------------------------------------------


async def test_an_idle_topic_ahead_in_the_order_does_not_starve_the_busy_one() -> None:
    idle, _idle_fake = _consumer(_INTENTS_DLQ, [], block=True)
    busy, busy_fake = _consumer(_EVENTS_DLQ, [900, 901])

    started = time.monotonic()
    result = await _handler({_INTENTS_DLQ: idle, _EVENTS_DLQ: busy}).run()
    elapsed = time.monotonic() - started

    assert result.total_processed == 2, (
        "the busy topic behind the idle one must still be drained"
    )
    assert _committed(busy_fake) == {(_EVENTS_DLQ, 0): 902}, busy_fake.commits
    assert elapsed < 2.0, (
        f"an empty topic must cost its idle probe (0.05s here), not the run's "
        f"5s wall clock; took {elapsed:.2f}s"
    )


async def test_the_wall_clock_is_shared_across_topics_not_multiplied_by_them() -> None:
    """OMN-17137 stays closed: a run returns inside max_run_duration_seconds
    however many topics are declared. Three blocked topics at 0.4s of probe
    each would exceed a 1.0s budget if the budget were per topic."""
    consumers = {
        topic: _consumer(topic, [], block=True)[0]
        for topic in (_EVENTS_DLQ, _INTENTS_DLQ, _COMMANDS_DLQ)
    }
    for consumer in consumers.values():
        object.__setattr__(
            consumer,
            "config",
            _config(
                consumer.config.dlq_topic,
                max_run_duration_seconds=1.0,
                idle_probe_seconds=0.4,
            ),
        )

    started = time.monotonic()
    result = await _handler(consumers).run()
    elapsed = time.monotonic() - started

    assert result.total_processed == 0, result
    assert elapsed < 1.6, (
        f"the run must respect ONE shared 1.0s budget across three topics, "
        f"not 0.4s per topic on top of it; took {elapsed:.2f}s"
    )


# --------------------------------------------------------------------------
# Starvation guard -- the order rotates.
# --------------------------------------------------------------------------


async def test_the_start_of_the_order_rotates_every_run() -> None:
    """Without rotation, a topic whose predecessor always fills its own record
    budget is never reached. That is this lane's steady state today: the events
    DLQ can supply max_records_per_run every single run, indefinitely."""
    handler = _handler(
        {
            topic: _consumer(topic, [], block=True)[0]
            for topic in (_EVENTS_DLQ, _INTENTS_DLQ, _COMMANDS_DLQ)
        }
    )

    starts = [(await handler.run()).dlq_topic for _ in range(4)]

    assert starts[:3] == [_EVENTS_DLQ, _INTENTS_DLQ, _COMMANDS_DLQ], starts
    assert starts[3] == _EVENTS_DLQ, f"the rotation must wrap: {starts}"


async def test_a_topic_that_fills_its_budget_does_not_consume_a_peers_budget() -> None:
    """The record budget is PER TOPIC. With max_records_per_run=3, a topic
    holding four records takes three and the next topic still gets its own."""
    greedy, greedy_fake = _consumer(_EVENTS_DLQ, [1, 2, 3, 4])
    peer, peer_fake = _consumer(_COMMANDS_DLQ, [50, 51])

    result = await _handler({_EVENTS_DLQ: greedy, _COMMANDS_DLQ: peer}).run()

    assert result.total_processed == 5, (
        f"3 from the greedy topic (its own budget) + 2 from the peer: {result}"
    )
    assert _committed(greedy_fake) == {(_EVENTS_DLQ, 0): 4}, greedy_fake.commits
    assert _committed(peer_fake) == {(_COMMANDS_DLQ, 0): 52}, peer_fake.commits


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_an_empty_consumer_mapping_is_refused() -> None:
    """A handler with no consumers would report clean bounded runs forever
    while draining nothing -- the same silence this ticket exists for."""
    with pytest.raises(ValueError, match="at least one DLQ consumer"):
        HandlerDlqReplay(
            consumers={},
            producer=_NoopReplayProducer(),  # type: ignore[arg-type]
            quarantine_producer=_RecordingQuarantineProducer(),  # type: ignore[arg-type]
            tracking=None,
        )

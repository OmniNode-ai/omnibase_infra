# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19085 -- a trigger with nothing to drain must not join the replay group.

Every DLQ record is a per-record TRIGGER for a whole drain, and every drain
started each declared topic's consumer. A start is a join of the shared
``onex-dlq-replay`` group, and a join into an Empty group waits the broker's
``group_initial_rebalance_delay`` (3000 ms on redpanda). Measured on the .201
dev lane 2026-09-24: the replay group's lag was 0 on all three topics while the
events trigger group held 44,178 stale triggers, each costing about 10 s. Its
committed offset moved once per ~69-record batch, every ~11.5 minutes, and the
broker readiness probe (300 s window) read ``group_not_synced`` for over a day.

The fix reads the replay group's undrained count first and skips the consumer
start of a topic at exactly 0. These tests pin the three things that make that
safe: only an explicit 0 skips, any failed or slow read drains everything as
before, and the probe's arithmetic counts an uncommitted partition from log
start rather than calling it drained.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

import pytest
from aiokafka import TopicPartition
from aiokafka.errors import KafkaConnectionError
from aiokafka.structs import OffsetAndMetadata

from omnibase_infra.nodes.node_dlq_replay_effect import engine_dlq_replay
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQ_REPLAY_CONSUMER_GROUP,
    DLQConsumer,
    DlqGroupBacklogProbe,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)

pytestmark = pytest.mark.unit

_ORIGINAL_TOPIC = "onex.evt.omniclaude.tool-executed.v1"  # onex-topic-allow: quotes the measured live record
_EVENTS_DLQ = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: the live DLQ this node drains
_INTENTS_DLQ = (
    "onex.dlq.omnibase-infra.intents.v1"  # onex-topic-allow: declared DLQ topic
)
_COMMANDS_DLQ = (
    "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: declared DLQ topic
)
_TOPICS = (_EVENTS_DLQ, _INTENTS_DLQ, _COMMANDS_DLQ)


def _config(topic: str, **overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": topic,
        "consumer_group": DLQ_REPLAY_CONSUMER_GROUP,
        "max_records_per_run": 3,
        "commit_every_n_records": 1000,
        "max_run_duration_seconds": 5.0,
        "idle_probe_seconds": 0.05,
        "backlog_probe_timeout_seconds": 0.2,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _payload() -> bytes:
    """Ineligible by retry count, so it quarantines without a replay publish."""
    return json.dumps(
        {
            "original_topic": _ORIGINAL_TOPIC,
            "original_message": {"key": "k", "value": '{"hello": "world"}'},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": 99,
        }
    ).encode("utf-8")


class _FakeRecord:
    def __init__(self, offset: int) -> None:
        self.value = _payload()
        self.offset = offset
        self.partition = 0


class _FakeAIOKafkaConsumer:
    def __init__(self, records: list[_FakeRecord]) -> None:
        self._records = records
        self._index = 0
        self.commits: list[Any] = []

    def __aiter__(self) -> _FakeAIOKafkaConsumer:
        return self

    async def __anext__(self) -> _FakeRecord:
        if not self._records:
            # A declared-but-empty topic: block until the idle probe gives up.
            await asyncio.Event().wait()
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record

    async def commit(self, offsets: Any = None) -> None:
        self.commits.append(offsets)

    async def stop(self) -> None:
        return None


class _CountingConsumer(DLQConsumer):
    """A real DLQConsumer whose start() is the group join this ticket avoids."""

    def __init__(self, topic: str, offsets: list[int]) -> None:
        super().__init__(_config(topic))
        self._offsets = offsets
        self.starts = 0

    async def start(self) -> None:
        self.starts += 1
        self._consumer = _FakeAIOKafkaConsumer(  # type: ignore[assignment]
            [_FakeRecord(o) for o in self._offsets]
        )
        self._started = True


class _NoopReplayProducer:
    def __init__(self) -> None:
        self._started = True

    async def start(self) -> None:  # pragma: no cover - already started
        return None

    async def stop(self) -> None:  # pragma: no cover - not owned by a lease
        return None

    async def replay_message(
        self, message: object, _cid: object
    ) -> None:  # pragma: no cover - records are ineligible
        return None


class _RecordingQuarantineProducer:
    def __init__(self) -> None:
        self._started = True
        self.count = 0

    async def start(self) -> None:  # pragma: no cover - already started
        return None

    async def stop(self) -> None:  # pragma: no cover - not owned by a lease
        return None

    async def quarantine_message(
        self, message: object, reason: str, _cid: object
    ) -> object:
        self.count += 1
        return object()

    async def quarantine_unparseable_record(
        self, record: object, _cid: object
    ) -> object:  # pragma: no cover - every record parses
        return object()


class _StaticProbe:
    def __init__(self, answer: Mapping[str, int]) -> None:
        self.answer = dict(answer)
        self.calls: list[tuple[str, ...]] = []

    async def undrained(self, topics: Sequence[str]) -> Mapping[str, int]:
        self.calls.append(tuple(topics))
        return self.answer


class _FailingProbe:
    async def undrained(self, topics: Sequence[str]) -> Mapping[str, int]:
        raise KafkaConnectionError("broker unreachable")


class _BrokenProbe:
    async def undrained(self, topics: Sequence[str]) -> Mapping[str, int]:
        raise AttributeError("a defect in the probe itself")


class _HangingProbe:
    async def undrained(self, topics: Sequence[str]) -> Mapping[str, int]:
        await asyncio.Event().wait()
        return {}  # pragma: no cover - never reached


def _consumers(
    offsets: Mapping[str, list[int]] | None = None,
) -> dict[str, _CountingConsumer]:
    offsets = offsets or {}
    return {
        topic: _CountingConsumer(topic, list(offsets.get(topic, [])))
        for topic in _TOPICS
    }


def _handler(
    consumers: Mapping[str, _CountingConsumer], probe: object | None
) -> tuple[HandlerDlqReplay, _RecordingQuarantineProducer]:
    quarantine = _RecordingQuarantineProducer()
    handler = HandlerDlqReplay(
        consumers=consumers,
        producer=_NoopReplayProducer(),  # type: ignore[arg-type]
        quarantine_producer=quarantine,  # type: ignore[arg-type]
        tracking=None,
        backlog_probe=probe,  # type: ignore[arg-type]
    )
    return handler, quarantine


@pytest.mark.asyncio
async def test_fully_committed_topics_are_not_joined() -> None:
    """RED on origin/dev: HandlerDlqReplay takes no backlog_probe, and every
    trigger starts all three consumers (three group joins) to drain nothing."""
    consumers = _consumers()
    probe = _StaticProbe(dict.fromkeys(_TOPICS, 0))
    handler, _ = _handler(consumers, probe)

    result = await handler.run()

    assert [c.starts for c in consumers.values()] == [0, 0, 0]
    assert result.total_processed == 0
    assert len(probe.calls) == 1
    assert set(probe.calls[0]) == set(_TOPICS)


@pytest.mark.asyncio
async def test_only_the_topic_with_a_backlog_is_joined_and_drained() -> None:
    consumers = _consumers({_COMMANDS_DLQ: [7, 8]})
    probe = _StaticProbe({_EVENTS_DLQ: 0, _INTENTS_DLQ: 0, _COMMANDS_DLQ: 2})
    handler, quarantine = _handler(consumers, probe)

    result = await handler.run()

    assert consumers[_EVENTS_DLQ].starts == 0
    assert consumers[_INTENTS_DLQ].starts == 0
    assert consumers[_COMMANDS_DLQ].starts == 1
    assert result.total_processed == 2
    assert quarantine.count == 2


@pytest.mark.asyncio
async def test_a_topic_the_probe_did_not_answer_for_is_drained() -> None:
    """An absent answer is not a zero: the topic drains as before."""
    consumers = _consumers({_INTENTS_DLQ: [3]})
    probe = _StaticProbe({_EVENTS_DLQ: 0, _COMMANDS_DLQ: 0})
    handler, _ = _handler(consumers, probe)

    result = await handler.run()

    assert consumers[_INTENTS_DLQ].starts == 1
    assert consumers[_EVENTS_DLQ].starts == 0
    assert consumers[_COMMANDS_DLQ].starts == 0
    assert result.total_processed == 1


@pytest.mark.asyncio
async def test_a_failed_probe_drains_every_topic() -> None:
    consumers = _consumers({_EVENTS_DLQ: [1]})
    handler, _ = _handler(consumers, _FailingProbe())

    result = await handler.run()

    assert [c.starts for c in consumers.values()] == [1, 1, 1]
    assert result.total_processed == 1


@pytest.mark.asyncio
async def test_an_unexpected_probe_error_never_fails_the_dispatch() -> None:
    """A raise out of run() would dead-letter the trigger onto the topic this
    node drains (the OMN-18084 amplifier); the run must drain instead."""
    consumers = _consumers({_EVENTS_DLQ: [1]})
    handler, _ = _handler(consumers, _BrokenProbe())

    result = await handler.run()

    assert [c.starts for c in consumers.values()] == [1, 1, 1]
    assert result.total_processed == 1


@pytest.mark.asyncio
async def test_a_hanging_probe_is_bounded_and_drains_every_topic() -> None:
    consumers = _consumers({_EVENTS_DLQ: [1]})
    handler, _ = _handler(consumers, _HangingProbe())

    started = time.monotonic()
    result = await handler.run()
    elapsed = time.monotonic() - started

    # 0.2 s probe bound + three 0.05 s idle probes, far inside the 5 s run bound.
    assert elapsed < 2.0, elapsed
    assert [c.starts for c in consumers.values()] == [1, 1, 1]
    assert result.total_processed == 1


@pytest.mark.asyncio
async def test_no_probe_keeps_the_previous_behaviour() -> None:
    consumers = _consumers()
    handler, _ = _handler(consumers, None)

    await handler.run()

    assert [c.starts for c in consumers.values()] == [1, 1, 1]


# --------------------------------------------------------------------------
# The probe's arithmetic, against faked Kafka clients.
# --------------------------------------------------------------------------


class _FakeAdmin:
    committed: dict[TopicPartition, OffsetAndMetadata] = {}
    instances: list[_FakeAdmin] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.group: str | None = None
        _FakeAdmin.instances.append(self)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def describe_topics(self, topics: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "topic": topic,
                "error_code": 0 if topic in _FakeReader.partitions else 3,
                "partitions": [
                    {"partition": p}
                    for p in sorted(_FakeReader.partitions.get(topic, ()))
                ],
            }
            for topic in topics
        ]

    async def list_consumer_group_offsets(
        self, group_id: str, partitions: list[TopicPartition] | None = None
    ) -> dict[TopicPartition, OffsetAndMetadata]:
        self.group = group_id
        return {tp: m for tp, m in self.committed.items() if tp in (partitions or [])}


class _FakeReader:
    partitions: dict[str, set[int]] = {}
    end: dict[TopicPartition, int] = {}
    begin: dict[TopicPartition, int] = {}
    instances: list[_FakeReader] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.stopped = False
        _FakeReader.instances.append(self)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True

    async def end_offsets(self, tps: list[TopicPartition]) -> dict[TopicPartition, int]:
        return {tp: self.end[tp] for tp in tps}

    async def beginning_offsets(
        self, tps: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {tp: self.begin[tp] for tp in tps}


@pytest.mark.asyncio
async def test_probe_counts_committed_uncommitted_and_unknown_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev0 = TopicPartition(_EVENTS_DLQ, 0)
    ev1 = TopicPartition(_EVENTS_DLQ, 1)
    cmd0 = TopicPartition(_COMMANDS_DLQ, 0)
    _FakeReader.partitions = {_EVENTS_DLQ: {0, 1}, _COMMANDS_DLQ: {0}}
    _FakeReader.end = {ev0: 100, ev1: 50, cmd0: 30}
    _FakeReader.begin = {ev0: 10, ev1: 20, cmd0: 5}
    _FakeReader.instances = []
    # ev0 fully committed; ev1 never committed (counts from log start);
    # cmd0 committed below log start (retention moved past it).
    _FakeAdmin.committed = {
        ev0: OffsetAndMetadata(100, ""),
        cmd0: OffsetAndMetadata(2, ""),
    }
    _FakeAdmin.instances = []
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaAdminClient", _FakeAdmin)
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaConsumer", _FakeReader)
    monkeypatch.setattr(engine_dlq_replay, "build_aiokafka_auth_kwargs_from_env", dict)

    probe = DlqGroupBacklogProbe(_config(_EVENTS_DLQ))
    answer = await probe.undrained(_TOPICS)

    assert answer == {_EVENTS_DLQ: 30, _COMMANDS_DLQ: 25}
    # The intents topic is unknown to the broker (error 3): omitted, never 0.
    assert _INTENTS_DLQ not in answer
    assert _FakeAdmin.instances[0].group == DLQ_REPLAY_CONSUMER_GROUP
    # The reader must never be a member of the group it measures.
    assert _FakeReader.instances[0].kwargs["group_id"] is None
    assert _FakeAdmin.instances[0].closed
    assert _FakeReader.instances[0].stopped


@pytest.mark.asyncio
async def test_probe_reports_zero_only_when_everything_is_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev0 = TopicPartition(_EVENTS_DLQ, 0)
    _FakeReader.partitions = {_EVENTS_DLQ: {0}}
    _FakeReader.end = {ev0: 21974002}
    _FakeReader.begin = {ev0: 21905923}
    _FakeAdmin.committed = {ev0: OffsetAndMetadata(21974002, "")}
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaAdminClient", _FakeAdmin)
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaConsumer", _FakeReader)
    monkeypatch.setattr(engine_dlq_replay, "build_aiokafka_auth_kwargs_from_env", dict)

    answer = await DlqGroupBacklogProbe(_config(_EVENTS_DLQ)).undrained([_EVENTS_DLQ])

    assert answer == {_EVENTS_DLQ: 0}

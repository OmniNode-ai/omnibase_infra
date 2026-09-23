# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19241 — a peer run must not stop a producer another run is using, and a
record whose handling keeps failing must stop being retried, loudly.

THE RACE. ``service_kernel`` builds ONE ``DLQProducer`` and ONE
``DLQQuarantineProducer`` and hands them to all three per-topic dispatchers,
while the OMN-18084 run mutex is keyed on the CONSUMER, and since OMN-18119
every topic has its own consumer. Two dispatchers draining two different topics
therefore run at the same time over the same producers. The run that found a
producer stopped starts it and stops it in its ``finally``; the run that found
it already started borrows it and stops nothing. When the starter finishes
first, the borrower's next quarantine raises ``RuntimeError("Quarantine
producer not started")``.

THE AMPLIFIER. That failure withholds the record's offset (OMN-17896), which is
correct. But the drain carried on past it and replayed every eligible record
behind it on the same partition. None of those offsets could be committed over
the blocked one, so the next run read and replayed all of them again, on every
run, for as long as the quarantine kept failing. Measured on the .201 dev lane
2026-09-23: one gate decision replayed about 22,800 times from 10:32:48Z.

HOW THE INTERLEAVING IS MADE DETERMINISTIC. The doubles subclass the real
engine classes and override only what touches Kafka, so the ``_started`` flag
that ``quarantine_message`` checks is the production one. Each drain waits on a
gate the test releases, which controls WHEN the drain body runs and nothing
else.

Evidence-Ticket: OMN-19241
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import pytest

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqDrainRecord,
)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

pytestmark = pytest.mark.unit

_EVENTS_DLQ = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: a declared subscribe topic of this node
_COMMANDS_DLQ = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: a declared subscribe topic of this node
_SOURCE_TOPIC = "onex.evt.omnimarket.prod-promotion-gate-evaluated.v1"  # onex-topic-allow: verbatim from the live storm
_OUTER_TIMEOUT_SECONDS = 5.0


def _config(topic: str) -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="test-broker:9092",
        dlq_topic=topic,
        max_replay_count=5,
        max_run_duration_seconds=2.0,
        idle_probe_seconds=0.2,
    )


def _message(*, offset: int, retry_count: int, partition: int = 0) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_SOURCE_TOPIC,
        original_key="k",
        original_value='{"decision": "gate"}',
        original_offset=str(offset),
        original_partition=0,
        failure_reason="handler dispatch failed",
        failure_timestamp="2026-09-23T10:32:40Z",
        correlation_id=uuid4(),
        retry_count=retry_count,
        error_type="InfraConnectionError",  # retryable: eligibility is the count
        dlq_offset=offset,
        dlq_partition=partition,
        raw_payload={"original_topic": _SOURCE_TOPIC},
    )


class _KafkaProducerStub:
    """Stands in for ``AIOKafkaProducer``; ``send_and_wait`` is the confirmation."""

    def __init__(self, sink: list[bytes]) -> None:
        self._sink = sink

    async def send_and_wait(self, topic: str, **kwargs: object) -> object:
        self._sink.append(cast("bytes", kwargs["value"]))
        return object()


class _QuarantineProducer(DLQQuarantineProducer):
    """The real quarantine producer, including its ``_started`` refusal."""

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        super().__init__(config)
        self.published: list[bytes] = []
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1
        self._producer = cast("AIOKafkaProducer", _KafkaProducerStub(self.published))
        self._started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._started = False
        self._producer = None


class _ReplayProducer(DLQProducer):
    """The real replay producer, including its ``_started`` refusal."""

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        super().__init__(config)
        self.published: list[bytes] = []
        self.start_calls = 0

    async def start(self) -> None:
        self.start_calls += 1
        self._producer = cast("AIOKafkaProducer", _KafkaProducerStub(self.published))
        self._started = True

    async def stop(self) -> None:
        self._started = False
        self._producer = None


class _GatedConsumer(DLQConsumer):
    """The real consumer lifecycle; each drain waits on a test-released gate."""

    def __init__(
        self, config: ModelDlqReplayEngineConfig, messages: list[ModelDlqMessage]
    ) -> None:
        super().__init__(config)
        self._messages = messages
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def start(self) -> None:
        self._consumer = cast("AIOKafkaConsumer", object())
        self._started = True

    async def stop(self) -> None:
        self._started = False
        self._consumer = None

    async def commit_offsets(self, offsets: Mapping[tuple[str, int], int]) -> None:
        return None

    async def consume_messages(self) -> AsyncIterator[DlqDrainRecord]:
        self.entered.set()
        await self.release.wait()
        for message in self._messages:
            yield message


@pytest.mark.unit
class TestSharedProducerLifecycle:
    """AC1 — a shared producer is never stopped while another run uses it."""

    @pytest.mark.asyncio
    async def test_a_peer_run_finishing_first_cannot_stop_the_shared_quarantine_producer(
        self,
    ) -> None:
        """RED at 533b19c23: the second run's quarantine raises.

        Run A drains the events DLQ and starts both shared producers. Run B
        drains the commands DLQ over a different consumer, so the per-consumer
        mutex does not serialise it, and borrows the producers A started. A
        finishes first and, at the parent commit, stops both in its
        ``finally``. B then reaches a record past ``max_replay_count`` and has
        to quarantine it.
        """
        producer = _ReplayProducer(_config(_EVENTS_DLQ))
        quarantine = _QuarantineProducer(_config(_EVENTS_DLQ))
        events_consumer = _GatedConsumer(_config(_EVENTS_DLQ), [])
        commands_consumer = _GatedConsumer(
            _config(_COMMANDS_DLQ), [_message(offset=7, retry_count=9)]
        )
        run_a = HandlerDlqReplay(
            consumers={_EVENTS_DLQ: events_consumer},
            producer=producer,
            quarantine_producer=quarantine,
        )
        run_b = HandlerDlqReplay(
            consumers={_COMMANDS_DLQ: commands_consumer},
            producer=producer,
            quarantine_producer=quarantine,
        )

        task_a = asyncio.create_task(run_a.run())
        await asyncio.wait_for(events_consumer.entered.wait(), _OUTER_TIMEOUT_SECONDS)
        task_b = asyncio.create_task(run_b.run())
        await asyncio.wait_for(commands_consumer.entered.wait(), _OUTER_TIMEOUT_SECONDS)

        events_consumer.release.set()
        await asyncio.wait_for(task_a, _OUTER_TIMEOUT_SECONDS)
        commands_consumer.release.set()
        result_b = await asyncio.wait_for(task_b, _OUTER_TIMEOUT_SECONDS)

        failures = [
            r.message for r in result_b.results if r.status == EnumReplayStatus.FAILED
        ]
        assert failures == [], (
            f"the peer run stopped the shared quarantine producer: {failures}"
        )
        assert result_b.quarantined == 1
        assert len(quarantine.published) == 1

    @pytest.mark.asyncio
    async def test_overlapping_runs_start_once_and_the_last_holder_stops(
        self,
    ) -> None:
        """The lease is not a leak: whoever holds the producer last stops it."""
        producer = _ReplayProducer(_config(_EVENTS_DLQ))
        quarantine = _QuarantineProducer(_config(_EVENTS_DLQ))
        events_consumer = _GatedConsumer(_config(_EVENTS_DLQ), [])
        commands_consumer = _GatedConsumer(_config(_COMMANDS_DLQ), [])
        runs = [
            HandlerDlqReplay(
                consumers={topic: consumer},
                producer=producer,
                quarantine_producer=quarantine,
            )
            for topic, consumer in (
                (_EVENTS_DLQ, events_consumer),
                (_COMMANDS_DLQ, commands_consumer),
            )
        ]

        tasks = [asyncio.create_task(run.run()) for run in runs]
        await asyncio.wait_for(events_consumer.entered.wait(), _OUTER_TIMEOUT_SECONDS)
        await asyncio.wait_for(commands_consumer.entered.wait(), _OUTER_TIMEOUT_SECONDS)
        assert quarantine.start_calls == 1
        assert producer.start_calls == 1

        events_consumer.release.set()
        await asyncio.wait_for(tasks[0], _OUTER_TIMEOUT_SECONDS)
        assert quarantine._started, "stopped while the second run still held it"

        commands_consumer.release.set()
        await asyncio.wait_for(tasks[1], _OUTER_TIMEOUT_SECONDS)
        assert not quarantine._started, "the last holder left the producer running"
        assert not producer._started
        assert quarantine.stop_calls == 1

    @pytest.mark.asyncio
    async def test_a_producer_started_by_the_caller_is_never_stopped_by_a_run(
        self,
    ) -> None:
        """``scripts/dlq_replay.py`` starts the producers itself and owns them."""
        producer = _ReplayProducer(_config(_EVENTS_DLQ))
        quarantine = _QuarantineProducer(_config(_EVENTS_DLQ))
        await producer.start()
        await quarantine.start()
        consumer = _GatedConsumer(_config(_EVENTS_DLQ), [])
        consumer.release.set()

        await asyncio.wait_for(
            HandlerDlqReplay(
                consumers={_EVENTS_DLQ: consumer},
                producer=producer,
                quarantine_producer=quarantine,
            ).run(),
            _OUTER_TIMEOUT_SECONDS,
        )

        assert quarantine._started
        assert producer._started
        assert quarantine.stop_calls == 0


class _CommittingConsumer:
    """One partition with a real committed offset, re-read from it each run."""

    def __init__(
        self, config: ModelDlqReplayEngineConfig, messages: list[ModelDlqMessage]
    ) -> None:
        self.config = config
        self._messages = messages
        self.committed = min(m.dlq_offset for m in messages)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def commit_offsets(self, offsets: Mapping[tuple[str, int], int]) -> None:
        for (_topic, _partition), next_offset in offsets.items():
            self.committed = next_offset

    async def consume_messages(self) -> AsyncIterator[DlqDrainRecord]:
        for message in self._messages:
            if message.dlq_offset >= self.committed:
                yield message


class _FlakyQuarantine:
    """Fails the first ``failures`` quarantine publishes, then confirms."""

    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.attempts = 0

    async def quarantine_message(
        self, message: ModelDlqMessage, reason: str, correlation_id: UUID
    ) -> object:
        self.attempts += 1
        if self.attempts <= self._failures:
            raise RuntimeError("Quarantine producer not started")
        return object()


class _RecordingReplay:
    def __init__(self) -> None:
        self.replayed: list[UUID] = []

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        self.replayed.append(message.correlation_id)


def _blocked_partition() -> list[ModelDlqMessage]:
    """Record N must be quarantined; N+1..N+3 are replay-eligible behind it."""
    return [
        _message(offset=10, retry_count=5),
        _message(offset=11, retry_count=0),
        _message(offset=12, retry_count=1),
        _message(offset=13, retry_count=2),
    ]


@pytest.mark.unit
class TestReplayCapBehindAFailedRecord:
    """AC2 and the loud bound on a record whose handling keeps failing."""

    @pytest.mark.asyncio
    async def test_records_behind_a_failed_quarantine_are_replayed_once_in_total(
        self,
    ) -> None:
        """RED at 533b19c23: N+1..N+3 are replayed on both runs, six publishes.

        Run 1 fails to quarantine N, so N's partition is blocked and nothing on
        it can be committed. Run 2 re-reads from N, quarantines it and replays
        the rest. Each of N+1..N+3 must reach the replay producer exactly once.
        """
        messages = _blocked_partition()
        consumer = _CommittingConsumer(_config(_EVENTS_DLQ), messages)
        replay = _RecordingReplay()
        handler = HandlerDlqReplay(
            consumers={_EVENTS_DLQ: cast("DLQConsumer", consumer)},
            producer=cast("DLQProducer", replay),
            quarantine_producer=cast(
                "DLQQuarantineProducer", _FlakyQuarantine(failures=1)
            ),
        )

        first = await asyncio.wait_for(handler.run(), _OUTER_TIMEOUT_SECONDS)
        second = await asyncio.wait_for(handler.run(), _OUTER_TIMEOUT_SECONDS)

        behind = [m.correlation_id for m in messages[1:]]
        assert sorted(map(str, replay.replayed)) == sorted(map(str, behind)), (
            "records behind the failed quarantine were replayed "
            f"{len(replay.replayed)} times for {len(behind)} records"
        )
        assert first.failed == 1
        assert first.completed == 0
        assert second.quarantined == 1
        assert second.completed == 3
        assert consumer.committed == 14

    @pytest.mark.asyncio
    async def test_a_record_that_keeps_failing_halts_its_partition_loudly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RED at 533b19c23: every run attempts N again, without end.

        After ``max_record_failure_attempts`` failed attempts on one record the
        drain stops attempting that partition, logs CRITICAL naming the
        coordinate, and reports it on every later run result.
        """
        messages = _blocked_partition()
        config = _config(_EVENTS_DLQ)
        consumer = _CommittingConsumer(config, messages)
        quarantine = _FlakyQuarantine(failures=1_000)
        replay = _RecordingReplay()
        handler = HandlerDlqReplay(
            consumers={_EVENTS_DLQ: cast("DLQConsumer", consumer)},
            producer=cast("DLQProducer", replay),
            quarantine_producer=cast("DLQQuarantineProducer", quarantine),
        )

        runs = config.max_record_failure_attempts + 3
        with caplog.at_level(logging.CRITICAL):
            results = [
                await asyncio.wait_for(handler.run(), _OUTER_TIMEOUT_SECONDS)
                for _ in range(runs)
            ]

        assert quarantine.attempts == config.max_record_failure_attempts
        assert replay.replayed == []
        assert consumer.committed == 10, "the unhandled record was committed past"
        halted = f"{_EVENTS_DLQ}/0@10"
        critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical) == 1
        assert halted in critical[0].getMessage()
        assert results[-1].halted_partitions == (halted,)
        assert results[0].halted_partitions == ()

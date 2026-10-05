# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20589 -- one runtime tick rejected by two subscribers settles at six copies.

The unit tests in ``tests/unit/nodes/node_dlq_replay_effect`` drive the handler
over typed messages. This file drives the real ``DLQConsumer``,
``DLQProducer`` and ``DLQQuarantineProducer`` with only the aiokafka objects
replaced, over dead letters in the JSON shape the auto-wired boundary wrote on
the dev-202 lane on 2026-10-05: ``original_message`` carrying key, value,
offset and partition, ``error_type`` ``HandlerDispatchFailureError`` and the
dispatcher named in ``failure_reason``.

The loop is closed the way the lane closed it. Every record the replay producer
publishes onto the tick topic gets a new source offset and is dead-lettered by
both failing subscribers, carrying the replay's ``x-replay-count`` header
forward as ``retry_count``. Measured live, one tick became 63 records
(1/2/4/8/16/32 by replay count). Fixed, it is one record per replay round.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

import pytest

from omnibase_infra.dlq import EnumReplayStatus
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)

pytestmark = pytest.mark.integration

_INTENTS_DLQ = "onex.dlq.omnibase-infra.intents.v1"  # onex-topic-allow: the DLQ the live dead letters landed on
_TICK_TOPIC = "onex.intent.platform.runtime-tick.v1"  # onex-topic-allow: verbatim from the live storm
_SUBSCRIBERS = (
    "dispatcher.auto.node_consumer_flow_prune_effect.HandlerConsumerFlowPrune",
    "dispatcher.auto.node_dead_letter_prune_effect.HandlerDeadLetterPrune",
)
_MAX_RUNS = 60
_RUN_TIMEOUT_SECONDS = 10.0


class _Record:
    def __init__(self, value: bytes, offset: int) -> None:
        self.value = value
        self.offset = offset
        self.partition = 0


class _Lane:
    """The tick topic and the intents DLQ, with two subscribers that always fail."""

    def __init__(self) -> None:
        self.tick_id = str(uuid4())
        self.tick_value = json.dumps(
            {
                "payload": {"tick_id": self.tick_id, "scheduler_id": "test"},
                "correlation_id": self.tick_id,
                "event_type": "platform.runtime-tick",
            }
        )
        self.tick_replay_counts: list[int] = []
        self.dlq: list[_Record] = []

    def deliver_tick(self, replay_count: int) -> None:
        source_offset = len(self.tick_replay_counts)
        self.tick_replay_counts.append(replay_count)
        for subscriber in _SUBSCRIBERS:
            dead_letter = {
                "correlation_id": self.tick_id,
                "error_type": "HandlerDispatchFailureError",
                "failure_class": None,
                "failure_reason": (
                    f"HandlerDispatchFailureError: dispatch to topic={_TICK_TOPIC} "
                    f"returned status=handler_error with no terminal output: "
                    f"Dispatcher '{subscriber}' failed: KeyError"
                ),
                "failure_timestamp": "2026-10-05T08:41:06.360484+00:00",
                "original_message": {
                    "key": self.tick_id,
                    "value": self.tick_value,
                    "offset": source_offset,
                    "partition": 0,
                },
                "original_topic": _TICK_TOPIC,
                "retry_count": replay_count,
                "validation_detail": None,
            }
            self.dlq.append(
                _Record(json.dumps(dead_letter).encode("utf-8"), len(self.dlq))
            )


class _AioConsumer:
    """Iterates the DLQ from the group's committed offset; sees appends live."""

    def __init__(self, lane: _Lane) -> None:
        self._lane = lane
        self.committed = 0
        self._index = 0

    def __aiter__(self) -> _AioConsumer:
        self._index = self.committed
        return self

    async def __anext__(self) -> _Record:
        if self._index >= len(self._lane.dlq):
            raise StopAsyncIteration
        record = self._lane.dlq[self._index]
        self._index += 1
        return record

    async def commit(self, offsets: Mapping[Any, int]) -> None:
        for next_offset in offsets.values():
            self.committed = next_offset


class _AioTickProducer:
    """A replay onto the tick topic is a new tick delivery; the cycle closes here."""

    def __init__(self, lane: _Lane) -> None:
        self._lane = lane
        self.sends: list[tuple[str, dict[str, bytes]]] = []

    async def send_and_wait(self, topic: str, **kwargs: Any) -> object:
        headers = dict(cast("list[tuple[str, bytes]]", kwargs["headers"]))
        self.sends.append((topic, headers))
        self._lane.deliver_tick(int(headers["x-replay-count"].decode("utf-8")))
        return object()


class _AioQuarantineProducer:
    def __init__(self) -> None:
        self.sends: list[str] = []

    async def send_and_wait(self, topic: str, **kwargs: Any) -> object:
        self.sends.append(topic)
        return object()


async def test_a_tick_two_subscribers_reject_settles_at_one_copy_per_round() -> None:
    lane = _Lane()
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_INTENTS_DLQ,
        max_replay_count=5,
        rate_limit_per_second=10_000.0,
        max_run_duration_seconds=5.0,
        idle_probe_seconds=0.05,
    )
    aio_consumer = _AioConsumer(lane)
    consumer = DLQConsumer(config)
    cast("Any", consumer)._consumer = aio_consumer
    consumer._started = True
    tick_producer = _AioTickProducer(lane)
    producer = DLQProducer(config)
    cast("Any", producer)._producer = tick_producer
    producer._started = True
    quarantine_sink = _AioQuarantineProducer()
    quarantine = DLQQuarantineProducer(config)
    cast("Any", quarantine)._producer = quarantine_sink
    quarantine._started = True
    handler = HandlerDlqReplay(
        consumers={_INTENTS_DLQ: consumer},
        producer=producer,
        quarantine_producer=quarantine,
        tracking=None,
    )

    lane.deliver_tick(0)
    statuses: list[EnumReplayStatus] = []
    for _ in range(_MAX_RUNS):
        if aio_consumer.committed >= len(lane.dlq):
            break
        result = await asyncio.wait_for(handler.run(), _RUN_TIMEOUT_SECONDS)
        statuses.extend(r.status for r in result.results)

    rounds = config.max_replay_count + 1
    assert aio_consumer.committed == len(lane.dlq), "the replay loop did not settle"
    assert lane.tick_replay_counts == list(range(rounds)), (
        f"one tick became {len(lane.tick_replay_counts)} copies on the tick topic"
    )
    assert [h["x-replay-count"] for _, h in tick_producer.sends] == [
        str(n).encode() for n in range(1, rounds)
    ]
    assert {topic for topic, _ in tick_producer.sends} == {_TICK_TOPIC}
    assert len(lane.dlq) == 2 * rounds
    assert statuses.count(EnumReplayStatus.COMPLETED) == config.max_replay_count
    assert statuses.count(EnumReplayStatus.SKIPPED) == config.max_replay_count
    assert statuses.count(EnumReplayStatus.QUARANTINED) == 2
    assert EnumReplayStatus.FAILED not in statuses
    assert len(quarantine_sink.sends) == 2


class _RecordingTracking:
    """Stands in for ``ServiceDlqTracking``: keeps every audit row's status."""

    def __init__(self) -> None:
        self.statuses: list[EnumReplayStatus] = []

    @property
    def is_tracking_enabled(self) -> bool:
        return True

    async def record_replay_attempt(self, record: Any) -> None:
        self.statuses.append(record.replay_status)


async def test_a_dry_run_after_a_live_replay_reports_the_sibling_and_records_nothing() -> (
    None
):
    """A dry run over the live shape writes no audit row, even for a duplicate.

    The live run replays the first dead letter of the tick, so its sibling is a
    duplicate. A dry run over the same consumer reports that sibling as a
    skipped duplicate and the next round's two dead letters as would-replay,
    and records, publishes and commits nothing (OMN-20589 review follow-up).
    """
    lane = _Lane()
    live_config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_INTENTS_DLQ,
        max_replay_count=5,
        rate_limit_per_second=10_000.0,
        max_run_duration_seconds=5.0,
        idle_probe_seconds=0.05,
        max_records_per_run=1,
    )
    aio_consumer = _AioConsumer(lane)
    consumer = DLQConsumer(live_config)
    cast("Any", consumer)._consumer = aio_consumer
    consumer._started = True
    tick_producer = _AioTickProducer(lane)
    producer = DLQProducer(live_config)
    cast("Any", producer)._producer = tick_producer
    producer._started = True
    quarantine_sink = _AioQuarantineProducer()
    quarantine = DLQQuarantineProducer(live_config)
    cast("Any", quarantine)._producer = quarantine_sink
    quarantine._started = True
    tracking = _RecordingTracking()

    lane.deliver_tick(0)
    live = HandlerDlqReplay(
        consumers={_INTENTS_DLQ: consumer},
        producer=producer,
        quarantine_producer=quarantine,
        tracking=cast("Any", tracking),
    )
    first = await asyncio.wait_for(live.run(), _RUN_TIMEOUT_SECONDS)
    assert [r.status for r in first.results] == [EnumReplayStatus.COMPLETED]
    assert tracking.statuses == [EnumReplayStatus.COMPLETED]
    assert aio_consumer.committed == 1
    assert len(lane.dlq) == 4

    consumer.config = live_config.model_copy(
        update={"dry_run": True, "max_records_per_run": 10}
    )
    dry = HandlerDlqReplay(
        consumers={_INTENTS_DLQ: consumer},
        producer=producer,
        quarantine_producer=quarantine,
        tracking=cast("Any", tracking),
    )
    second = await asyncio.wait_for(dry.run(), _RUN_TIMEOUT_SECONDS)

    assert [r.status for r in second.results] == [
        EnumReplayStatus.SKIPPED,
        EnumReplayStatus.PENDING,
        EnumReplayStatus.PENDING,
    ]
    assert second.results[0].message.startswith("DRY RUN")
    assert tracking.statuses == [EnumReplayStatus.COMPLETED], (
        "the dry run wrote an audit row"
    )
    assert len(tick_producer.sends) == 1
    assert quarantine_sink.sends == []
    assert aio_consumer.committed == 1

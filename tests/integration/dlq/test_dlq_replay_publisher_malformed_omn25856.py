# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25856 -- a publisher_malformed dead letter is quarantined, never replayed.

Drives the real ``DLQConsumer``, ``DLQProducer``, ``DLQQuarantineProducer`` and
``HandlerDlqReplay`` with only the aiokafka objects replaced, over dead letters
in the JSON shape the dead-letter publisher writes. A ``publisher_malformed``
record must reach the quarantine producer and never the replay producer; a
transient failure on the same topic is still replayed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

import pytest

from omnibase_infra.dlq import EnumReplayStatus
from omnibase_infra.enums import EnumDlqFailureClass
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

_COMMAND_TOPIC = (
    "onex.cmd.omnimarket.delegate-skill.v1"  # onex-topic-allow: live topic measured
)
_DLQ_TOPIC = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: live DLQ


class _Record:
    def __init__(self, value: bytes, offset: int) -> None:
        self.value = value
        self.offset = offset
        self.partition = 0


def _dead_letter(failure_class: str, offset: int) -> _Record:
    envelope = {
        "correlation_id": str(uuid4()),
        "error_type": "HandlerDispatchFailureError",
        "failure_class": failure_class,
        "failure_reason": f"dispatch failed ({failure_class})",
        "failure_timestamp": "2026-11-24T17:00:00+00:00",
        "original_message": {
            "key": str(uuid4()),
            "value": json.dumps({"task_type": "x"}),
            "offset": offset,
            "partition": 0,
        },
        "original_topic": _COMMAND_TOPIC,
        "retry_count": 0,
        "validation_detail": None,
    }
    return _Record(json.dumps(envelope).encode("utf-8"), offset)


class _AioConsumer:
    def __init__(self, records: list[_Record]) -> None:
        self._records = records
        self.committed = 0
        self._index = 0

    def __aiter__(self) -> _AioConsumer:
        self._index = self.committed
        return self

    async def __anext__(self) -> _Record:
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record

    async def commit(self, offsets: Mapping[Any, int]) -> None:
        for next_offset in offsets.values():
            self.committed = next_offset


class _AioSink:
    def __init__(self) -> None:
        self.sends: list[str] = []

    async def send_and_wait(self, topic: str, **kwargs: Any) -> object:
        self.sends.append(topic)
        return object()


async def test_publisher_malformed_is_quarantined_and_transient_is_replayed() -> None:
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_DLQ_TOPIC,
        max_replay_count=5,
        rate_limit_per_second=10_000.0,
        max_run_duration_seconds=5.0,
        idle_probe_seconds=0.05,
    )
    records = [
        _dead_letter(EnumDlqFailureClass.PUBLISHER_MALFORMED.value, 0),
        _dead_letter(EnumDlqFailureClass.CONSUMER_ERROR.value, 1),
    ]
    aio_consumer = _AioConsumer(records)
    consumer = DLQConsumer(config)
    cast("Any", consumer)._consumer = aio_consumer
    consumer._started = True
    replay_sink = _AioSink()
    producer = DLQProducer(config)
    cast("Any", producer)._producer = replay_sink
    producer._started = True
    quarantine_sink = _AioSink()
    quarantine = DLQQuarantineProducer(config)
    cast("Any", quarantine)._producer = quarantine_sink
    quarantine._started = True
    handler = HandlerDlqReplay(
        consumers={_DLQ_TOPIC: consumer},
        producer=producer,
        quarantine_producer=quarantine,
        tracking=None,
    )

    result = await asyncio.wait_for(handler.run(), 10.0)

    assert [r.status for r in result.results] == [
        EnumReplayStatus.QUARANTINED,
        EnumReplayStatus.COMPLETED,
    ]
    assert replay_sink.sends == [_COMMAND_TOPIC]
    assert len(quarantine_sink.sends) == 1
    assert aio_consumer.committed == 2

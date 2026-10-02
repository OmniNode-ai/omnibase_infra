# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20318 -- the replay node drains the events DLQ past gateway forensic records.

The unit tests in ``tests/unit/nodes/node_dlq_replay_effect`` pin the record
shape. This file drives the real ``DLQConsumer`` and ``HandlerDlqReplay`` over a
mixed events-DLQ backlog in the order measured on the dev broker 2026-10-02: the
gateway's own quarantine records (no ``original_message``) interleaved with
replayable records. Only the aiokafka consumer and the producers are replaced.
The gateway records are skipped, nothing reaches the quarantine sink, the
replayable records replay, and the committed offset passes the whole backlog.
"""

from __future__ import annotations

import json
from typing import Any, cast
from uuid import uuid4

import pytest

from omnibase_infra.dlq import EnumReplayStatus
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)

pytestmark = pytest.mark.integration

_EVENTS_DLQ = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: a declared subscribe topic of this node
_DENIED = "onex.evt.omniclaude.skill-started.v1"  # onex-topic-allow: verbatim from the live denial


def _replayable() -> bytes:
    return json.dumps(
        {
            "original_topic": _DENIED,
            "original_message": {"key": "k", "value": '{"hello": "world"}'},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": 0,
        }
    ).encode("utf-8")


def _gateway_forensic(offset: int) -> bytes:
    return json.dumps(
        {
            "original_topic": _DENIED,
            "original_partition": 0,
            "original_offset": offset,
            "direction": "outbound",
            "failure_class": "gateway_egress_denied_record",
            "error_type": "GatewayEgressDeniedError",
            "error_message": "GatewayEgressDeniedError: [REDACTED]",
        }
    ).encode("utf-8")


class _Record:
    def __init__(self, value: bytes, offset: int) -> None:
        self.value = value
        self.offset = offset
        self.partition = 0


class _Consumer:
    def __init__(self, records: list[_Record]) -> None:
        self._records = records
        self._index = 0
        self.position = 0
        self.commits: list[Any] = []

    def __aiter__(self) -> _Consumer:
        return self

    async def __anext__(self) -> _Record:
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        self.position = record.offset + 1
        return record

    async def commit(self, offsets: Any = None) -> None:
        self.commits.append(self.position if offsets is None else offsets)


class _Producer:
    def __init__(self) -> None:
        self._started = True
        self.replayed: list[object] = []
        self.quarantined: list[object] = []

    async def replay_message(self, message: object, _cid: object) -> None:
        self.replayed.append(message)

    async def quarantine_message(
        self, message: object, _reason: str, _cid: object
    ) -> object:
        self.quarantined.append(message)
        return object()

    async def quarantine_unparseable_record(
        self, record: object, _cid: object
    ) -> object:
        self.quarantined.append(record)
        return object()


def _max_committed(commits: list[Any]) -> int:
    highest = 0
    for entry in commits:
        if isinstance(entry, int):
            highest = max(highest, entry)
        elif isinstance(entry, dict):
            for value in entry.values():
                highest = max(highest, int(getattr(value, "offset", value)))
    return highest


async def test_events_dlq_backlog_with_gateway_records_drains_without_requarantine() -> (
    None
):
    payloads = [_gateway_forensic(i) if i % 3 else _replayable() for i in range(30)]
    forensic = sum(1 for i in range(30) if i % 3)
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_EVENTS_DLQ,
        max_records_per_run=100,
        commit_every_n_records=1000,
        max_run_duration_seconds=5.0,
    )
    consumer = DLQConsumer(config)
    fake = _Consumer([_Record(p, i) for i, p in enumerate(payloads)])
    cast("Any", consumer)._consumer = fake
    consumer._started = True
    producer = _Producer()
    quarantine = _Producer()

    result = await HandlerDlqReplay(
        consumers={_EVENTS_DLQ: consumer},
        producer=cast("Any", producer),
        quarantine_producer=cast("Any", quarantine),
        tracking=None,
    ).run()

    assert quarantine.quarantined == [], quarantine.quarantined
    assert result.quarantined == 0, result
    assert result.failed == 0, result
    assert result.total_processed == 30, result
    assert len(producer.replayed) == 30 - forensic
    assert (
        sum(1 for r in result.results if r.status == EnumReplayStatus.SKIPPED)
        == forensic
    )
    assert _max_committed(fake.commits) >= 30, fake.commits

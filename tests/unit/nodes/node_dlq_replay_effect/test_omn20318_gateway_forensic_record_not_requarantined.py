# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20318 -- a gateway quarantine record is already quarantined; the replay
node does not quarantine it a second time.

Measured read-only on the dev lane broker 2026-10-02 (dlq-depth-monitor run
36996632022, then a two-hour sample of ``onex.dlq.omnibase-infra.events.v1``
and ``onex.dlq.omnibase-infra.quarantine.v1``): the quarantine sink took 15
arrivals in the 30-minute window, and 147 of the 149 records sampled over two
hours were ``quarantine_class=unparseable_dlq_record`` written by
``node_dlq_replay_effect`` for records read from the events DLQ, with the reason
``original_message has no 'value' key``. The events DLQ held 145 records over
the same two hours.

Cause: ``NodeGatewayDelivery._quarantine_undecodable_message`` dead-letters a
record the gateway cannot deliver (an egress denial, a tenant refusal, an
undecodable body) onto the DLQ topic derived from the record's own topic, in the
gateway's own forensic shape -- ``original_topic``, ``direction``,
``failure_class=gateway_<class>_record``, ``error_type``, ``error_message`` --
with no ``original_message``. That record IS the gateway's durable quarantine of
the delivery failure. ``node_dlq_replay_effect`` drains the same DLQ topic,
cannot establish a body from a record that never carried one, and quarantined
the gateway's quarantine record a second time, onto the terminal quarantine sink.

The record stays on the events DLQ, which is where the gateway put it; the
replay node completes its offset past it without publishing and without
replaying, and says so in the run result.
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

pytestmark = pytest.mark.unit

_EVENTS_DLQ = "events-dlq-fixture"
_DENIED_TOPIC = "denied-topic-fixture"


def _config() -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_EVENTS_DLQ,
        max_records_per_run=50,
        commit_every_n_records=1000,
        max_run_duration_seconds=5.0,
    )


def _valid_dlq_payload() -> bytes:
    return json.dumps(
        {
            "original_topic": _DENIED_TOPIC,
            "original_message": {"key": "k", "value": '{"hello": "world"}'},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": 0,
        }
    ).encode("utf-8")


def _gateway_forensic_payload(
    failure_class: str = "gateway_egress_denied_record",
    error_type: str = "GatewayEgressDeniedError",
    direction: str = "outbound",
) -> bytes:
    """The shape ``_build_quarantine_payload`` writes, as measured live."""
    return json.dumps(
        {
            "original_topic": _DENIED_TOPIC,
            "original_partition": 0,
            "original_offset": 41,
            "direction": direction,
            "failure_class": failure_class,
            "error_type": error_type,
            "error_message": f"{error_type}: [REDACTED - potentially sensitive data]",
        }
    ).encode("utf-8")


def _absent_value_dlq_payload() -> bytes:
    """A record that is NOT the gateway's: no ``original_message.value``."""
    return json.dumps(
        {
            "original_topic": _DENIED_TOPIC,
            "original_message": {"key": "k"},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": 0,
        }
    ).encode("utf-8")


class _FakeConsumerRecord:
    def __init__(self, value: bytes | None, offset: int, partition: int = 0) -> None:
        self.value = value
        self.offset = offset
        self.partition = partition


class _FakeAIOKafkaConsumer:
    def __init__(self, records: list[_FakeConsumerRecord]) -> None:
        self._records = records
        self._index = 0
        self.position = 0
        self.commits: list[Any] = []

    def __aiter__(self) -> _FakeAIOKafkaConsumer:
        return self

    async def __anext__(self) -> _FakeConsumerRecord:
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        self.position = record.offset + 1
        return record

    async def commit(self, offsets: Any = None) -> None:
        self.commits.append(self.position if offsets is None else offsets)

    async def stop(self) -> None:
        raise AssertionError("not reached")


class _NoopReplayProducer:
    def __init__(self) -> None:
        self._started = True
        self.replayed: list[object] = []

    async def start(self) -> None:
        raise AssertionError("not reached")

    async def stop(self) -> None:
        raise AssertionError("not reached")

    async def replay_message(self, message: object, _cid: object) -> None:
        self.replayed.append(message)


class _RecordingQuarantineProducer:
    def __init__(self) -> None:
        self._started = True
        self.calls: list[tuple[str, object]] = []

    async def start(self) -> None:
        raise AssertionError("not reached")

    async def stop(self) -> None:
        raise AssertionError("not reached")

    async def quarantine_message(
        self, message: object, reason: str, _cid: object
    ) -> object:
        self.calls.append(("parsed", reason))
        return object()

    async def quarantine_unparseable_record(
        self, record: object, _cid: object
    ) -> object:
        self.calls.append(("unparseable", getattr(record, "reason", "")))
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


async def _drain(
    payloads: list[bytes],
) -> tuple[
    Any, _RecordingQuarantineProducer, _NoopReplayProducer, _FakeAIOKafkaConsumer
]:
    config = _config()
    consumer = DLQConsumer(config)
    fake = _FakeAIOKafkaConsumer(
        [_FakeConsumerRecord(p, offset) for offset, p in enumerate(payloads)]
    )
    cast("Any", consumer)._consumer = fake
    consumer._started = True
    producer = _NoopReplayProducer()
    quarantine = _RecordingQuarantineProducer()
    handler = HandlerDlqReplay(
        consumers={config.dlq_topic: consumer},
        producer=cast("Any", producer),
        quarantine_producer=cast("Any", quarantine),
        tracking=None,
    )
    result = await handler.run()
    return result, quarantine, producer, fake


@pytest.mark.parametrize(
    ("failure_class", "error_type"),
    [
        ("gateway_egress_denied_record", "GatewayEgressDeniedError"),
        ("gateway_refused_record", "GatewayRecordRefusedError"),
        ("gateway_undecodable_record", "ValueError"),
    ],
)
async def test_a_gateway_forensic_record_is_not_quarantined_a_second_time(
    failure_class: str, error_type: str
) -> None:
    """RED on origin/dev: the record at offset 1 is published onto the
    quarantine sink as ``unparseable_dlq_record`` ("has no 'value' key")."""
    result, quarantine, producer, fake = await _drain(
        [
            _valid_dlq_payload(),
            _gateway_forensic_payload(failure_class, error_type),
            _valid_dlq_payload(),
        ]
    )

    assert quarantine.calls == [], (
        "the gateway's own quarantine record was published onto the "
        f"quarantine sink again: {quarantine.calls}"
    )
    assert result.quarantined == 0, result
    assert result.failed == 0, result
    assert len(producer.replayed) == 2, "both real DLQ records still replay"
    assert result.total_processed == 3, result
    skipped = [r for r in result.results if r.status == EnumReplayStatus.SKIPPED]
    assert len(skipped) == 1, [r.status for r in result.results]
    assert failure_class in skipped[0].message, skipped[0].message
    assert _max_committed(fake.commits) >= 3, (
        "the offset must advance past the forensic record, or it is redelivered "
        f"every run: commits={fake.commits}"
    )


async def test_a_record_that_is_not_the_gateways_is_still_quarantined() -> None:
    """Positive control. A record with no ``original_message.value`` and no
    gateway forensic shape keeps taking the OMN-17896 quarantine path; an
    implementation that skips every record without a body passes the test above
    and silently drops the very records OMN-17896 exists to keep."""
    result, quarantine, _producer, fake = await _drain(
        [_valid_dlq_payload(), _absent_value_dlq_payload()]
    )

    assert [kind for kind, _ in quarantine.calls] == ["unparseable"], quarantine.calls
    assert result.quarantined == 1, result
    assert _max_committed(fake.commits) >= 2, fake.commits


async def test_a_gateway_class_name_without_the_forensic_shape_is_not_skipped() -> None:
    """A record that merely carries a ``gateway_`` failure_class but also a
    parseable ``original_message`` is an ordinary DLQ record and replays."""
    payload = json.dumps(
        {
            "original_topic": _DENIED_TOPIC,
            "original_message": {"key": "k", "value": '{"hello": "world"}'},
            "failure_class": "gateway_egress_denied_record",
            "correlation_id": str(uuid4()),
            "error_type": "GatewayEgressDeniedError",
            "retry_count": 0,
        }
    ).encode("utf-8")

    result, quarantine, producer, _fake = await _drain([payload])

    assert quarantine.calls == [], quarantine.calls
    assert len(producer.replayed) == 1, "an ordinary record is replayed, not skipped"
    assert not [r for r in result.results if r.status == EnumReplayStatus.SKIPPED]

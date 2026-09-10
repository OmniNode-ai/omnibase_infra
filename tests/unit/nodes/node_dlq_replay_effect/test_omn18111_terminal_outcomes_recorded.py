# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18111 -- "records every terminal outcome in ``dlq_replay_history``" is a
claim the node makes three times and honours zero times.

The claim appears in ``contract.yaml``'s ``description``, in
``handler_dlq_replay.py``'s module docstring (as a truthfulness invariant) and
in ``node.py``. ``dlq_replay_history`` has never held a row, against ~62,000
terminal outcomes in one 18-minute window on the .201 dev lane (OMN-18084
residual (3)).

Two independent gaps produce that zero, and both are covered here:

  * The runtime never provides the dependency at all --
    ``_build_runtime_handler_dependencies`` builds
    ``dependencies["HandlerDlqReplay"]`` with ``consumer`` / ``producer`` /
    ``quarantine_producer`` and no ``tracking``, so ``_record()`` returns at
    its first line on every call. That half is gated in
    ``tests/integration/test_omn18111_dlq_tracking_dependency_wired.py``,
    beside the resolver it is a fact about.

  * ``_quarantine_unparseable`` never calls the recorder AT ALL, so even with
    the dependency wired the OMN-17896 path -- the records whose original body
    could not be established, the ones most in need of an audit row -- would
    still write nothing. That is test 2 below.

Test 3 is the constraint the repair must not violate. The audit write is a side
effect of an ALREADY-DURABLE outcome: the quarantine publish is confirmed before
``_record`` is reached. If a tracking failure were allowed to propagate it would
turn a confirmed quarantine into a ``FAILED`` result, block the partition, and
have the record redelivered and re-quarantined -- re-amplifying the exact
OMN-18084 loop this node was just repaired for. A database being down must cost
an audit row, never a durable outcome.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.dlq.models.model_dlq_replay_record import ModelDlqReplayRecord
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    ModelUnparseableDlqRecord,
)

pytestmark = pytest.mark.unit

_ORIGINAL_TOPIC = "onex.evt.omniclaude.tool-executed.v1"  # onex-topic-allow: quotes the measured live record
_DLQ_TOPIC = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: the live DLQ this node drains
_QUARANTINE_TOPIC = "onex.dlq.omnibase-infra.quarantine.v1"  # onex-topic-allow: the terminal sink this node publishes to


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": _DLQ_TOPIC,
        "max_records_per_run": 50,
        "commit_every_n_records": 1000,
        "max_run_duration_seconds": 5.0,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _dlq_payload(*, retry_count: int = 0) -> bytes:
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

    async def stop(self) -> None:  # pragma: no cover - not reached
        return None


def _wired_consumer(
    config: ModelDlqReplayEngineConfig, records: list[_FakeConsumerRecord]
) -> tuple[DLQConsumer, _FakeAIOKafkaConsumer]:
    consumer = DLQConsumer(config)
    fake = _FakeAIOKafkaConsumer(records)
    consumer._consumer = fake  # type: ignore[assignment]
    consumer._started = True
    return consumer, fake


class _NoopReplayProducer:
    def __init__(self) -> None:
        self._started = True
        self.replayed: list[object] = []

    async def start(self) -> None:  # pragma: no cover - not reached
        return None

    async def stop(self) -> None:  # pragma: no cover - not reached
        return None

    async def replay_message(self, message: object, _cid: object) -> None:
        self.replayed.append(message)


class _RecordingQuarantineProducer:
    def __init__(self) -> None:
        self._started = True
        self.calls: list[tuple[str, object]] = []

    async def start(self) -> None:  # pragma: no cover - not reached
        return None

    async def stop(self) -> None:  # pragma: no cover - not reached
        return None

    async def quarantine_message(
        self, message: object, reason: str, _cid: object
    ) -> object:
        self.calls.append(("parsed", message))
        return object()

    async def quarantine_unparseable_record(
        self, record: ModelUnparseableDlqRecord, _cid: object
    ) -> object:
        self.calls.append(("unparseable", record))
        return object()


class _RecordingTracking:
    """Stands in for ``ServiceDlqTracking``.

    ``raises`` reproduces a database that is down: ``record_replay_attempt``
    declares ``InfraConnectionError`` / ``InfraTimeoutError`` /
    ``InfraUnavailableError`` / ``RuntimeHostError`` and its circuit breaker
    raises on its own once the failure threshold is crossed.
    """

    def __init__(self, *, raises: bool = False, enabled: bool = True) -> None:
        self.records: list[ModelDlqReplayRecord] = []
        self._raises = raises
        self._enabled = enabled

    @property
    def is_tracking_enabled(self) -> bool:
        return self._enabled

    async def record_replay_attempt(self, record: ModelDlqReplayRecord) -> None:
        self.records.append(record)
        if self._raises:
            raise RuntimeError("dlq_replay_history is unreachable")


def _handler(
    consumer: DLQConsumer,
    producer: _NoopReplayProducer,
    quarantine: _RecordingQuarantineProducer,
    tracking: _RecordingTracking | None,
) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumer=consumer,
        producer=producer,  # type: ignore[arg-type]
        quarantine_producer=quarantine,  # type: ignore[arg-type]
        tracking=tracking,  # type: ignore[arg-type]
    )


def _max_committed(commits: list[Any]) -> int:
    highest = 0
    for commit in commits:
        if isinstance(commit, dict):
            for offset in commit.values():
                highest = max(highest, int(offset))
        else:
            highest = max(highest, int(commit))
    return highest


# --------------------------------------------------------------------------
# Test 1 (AC2) -- every PARSEABLE terminal outcome reaches the recorder.
# Green on origin/dev for the handler in isolation; it is the regression guard
# that keeps the parseable half honest once the runtime actually supplies a
# tracking service.
# --------------------------------------------------------------------------


async def test_1_every_parseable_terminal_outcome_is_recorded_exactly_once() -> None:
    config = _config()
    consumer, _fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(_dlq_payload(retry_count=0), 10),
            _FakeConsumerRecord(_dlq_payload(retry_count=99), 11),
        ],
    )
    tracking = _RecordingTracking()
    result = await _handler(
        consumer, _NoopReplayProducer(), _RecordingQuarantineProducer(), tracking
    ).run()

    assert result.total_processed == 2, result
    assert result.completed == 1, result
    assert result.quarantined == 1, result
    assert len(tracking.records) == 2, (
        "one audit row per terminal outcome, no more and no fewer -- "
        f"got {[r.replay_status for r in tracking.records]}"
    )
    statuses = {r.replay_status for r in tracking.records}
    assert statuses == {EnumReplayStatus.COMPLETED, EnumReplayStatus.QUARANTINED}
    offsets = sorted(r.dlq_offset for r in tracking.records)
    assert offsets == [10, 11], offsets


# --------------------------------------------------------------------------
# Test 2 (AC3) -- RED on origin/dev: the unparseable path records NOTHING.
# --------------------------------------------------------------------------


async def test_2_unparseable_quarantine_is_recorded_with_its_dlq_coordinate() -> None:
    """RED on origin/dev: ``_quarantine_unparseable`` returns without ever
    calling ``_record``, so ``tracking.records`` is empty while a record was
    durably quarantined."""
    config = _config()
    consumer, _fake = _wired_consumer(
        config,
        [_FakeConsumerRecord(b"{not json at all", 7, partition=3)],
    )
    quarantine = _RecordingQuarantineProducer()
    tracking = _RecordingTracking()

    result = await _handler(consumer, _NoopReplayProducer(), quarantine, tracking).run()

    assert result.quarantined == 1, result
    assert [kind for kind, _ in quarantine.calls] == ["unparseable"], quarantine.calls
    assert len(tracking.records) == 1, (
        "a durably quarantined unparseable record is a TERMINAL outcome and the "
        "contract says every terminal outcome is recorded"
    )
    row = tracking.records[0]
    assert row.replay_status == EnumReplayStatus.QUARANTINED, row
    assert row.success is False, row
    assert row.dlq_offset == 7, row
    assert row.dlq_partition == 3, row
    assert row.original_topic == _DLQ_TOPIC, (
        "an unparseable record has no readable original topic; the only true "
        "topic is the DLQ it was read from"
    )
    assert row.target_topic == _QUARANTINE_TOPIC, (
        "it went to quarantine, and the row must say so rather than name a "
        "replay target that never existed"
    )
    assert row.error_message and "not JSON" in row.error_message, row.error_message
    assert isinstance(row.original_message_id, UUID)


async def test_2b_unparseable_dlq_coordinate_id_is_deterministic() -> None:
    """The same DLQ coordinate yields the same ``original_message_id`` twice.

    An unparseable record carries no correlation id, so the audit row needs an
    identity derived from something real. The DLQ coordinate is that something:
    two rows for one coordinate are recognisably the same record, which is the
    identity a reclassification owner actually needs.
    """
    ids: list[UUID] = []
    for _ in range(2):
        config = _config()
        consumer, _fake = _wired_consumer(
            config, [_FakeConsumerRecord(b"{not json at all", 7, partition=3)]
        )
        tracking = _RecordingTracking()
        await _handler(
            consumer, _NoopReplayProducer(), _RecordingQuarantineProducer(), tracking
        ).run()
        ids.append(tracking.records[0].original_message_id)

    assert ids[0] == ids[1], ids


# --------------------------------------------------------------------------
# Test 3 (AC4) -- RED on origin/dev once test 2's call exists: a failing audit
# write must not un-durable a confirmed quarantine.
# --------------------------------------------------------------------------


async def test_3_a_failing_audit_write_never_changes_the_verdict() -> None:
    """RED without failure isolation: the raise propagates out of ``_quarantine``,
    out of ``run()``, the batch aborts with nothing committed, and the record is
    redelivered and re-quarantined -- the OMN-18084 amplification, reintroduced
    through the audit path."""
    config = _config()
    consumer, fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(_dlq_payload(retry_count=99), 20),
            _FakeConsumerRecord(_dlq_payload(retry_count=0), 21),
        ],
    )
    quarantine = _RecordingQuarantineProducer()
    tracking = _RecordingTracking(raises=True)

    result = await _handler(consumer, _NoopReplayProducer(), quarantine, tracking).run()

    assert result.total_processed == 2, result
    assert result.quarantined == 1, result
    assert result.completed == 1, result
    assert result.failed == 0, (
        "the quarantine publish was CONFIRMED; a failed audit row does not "
        "make it undurable"
    )
    assert _max_committed(fake.commits) >= 22, (
        f"both records completed and their offsets must commit: {fake.commits}"
    )


async def test_3b_absent_tracking_is_still_a_working_drain() -> None:
    """``tracking`` is ``required: false`` in the contract, and a runtime with
    no database must still drain."""
    config = _config()
    consumer, fake = _wired_consumer(
        config, [_FakeConsumerRecord(_dlq_payload(retry_count=99), 30)]
    )
    result = await _handler(
        consumer, _NoopReplayProducer(), _RecordingQuarantineProducer(), None
    ).run()

    assert result.quarantined == 1, result
    assert _max_committed(fake.commits) >= 31, fake.commits


async def test_3c_disabled_tracking_service_writes_nothing() -> None:
    """An uninitialised service reports ``is_tracking_enabled`` False and must
    be treated as absent rather than called."""
    config = _config()
    consumer, _fake = _wired_consumer(
        config, [_FakeConsumerRecord(_dlq_payload(retry_count=99), 40)]
    )
    tracking = _RecordingTracking(enabled=False)
    result = await _handler(
        consumer, _NoopReplayProducer(), _RecordingQuarantineProducer(), tracking
    ).run()

    assert result.quarantined == 1, result
    assert tracking.records == [], tracking.records

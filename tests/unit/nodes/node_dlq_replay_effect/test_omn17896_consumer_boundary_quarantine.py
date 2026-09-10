# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17896 -- the consumer boundary: a refusal must reach quarantine, and a
commit must name only the records that completed.

WHY THE PARSE REFUSAL ALONE WOULD HAVE BEEN A STALL, NOT A REPAIR. The parse
runs inside ``DLQConsumer.consume_messages``' generator, in the ``yield``
expression; the enclosing ``try`` catches only ``asyncio.CancelledError`` and
``KafkaError``. Quarantine lives downstream in ``HandlerDlqReplay._quarantine``,
and the acquisition in ``run()`` -- ``await asyncio.wait_for(anext(messages),
timeout=remaining)`` -- guards only ``StopAsyncIteration`` and ``TimeoutError``.
So a typed refusal raised at the yield propagates out of ``anext``, past the
quarantine decision, and out of ``run()``: the trailing commit and the summary
are both skipped, the batch aborts, nothing is quarantined, and the next run
re-reads the same record forever. This was already LIVE before this ticket --
``_parse_retry_count`` raises ``ValueError`` on that same path.

THREE FURTHER DROPS ON THE SAME SEAM, ALL LIVE ON ``origin/dev``:

  * The generator ``continue``d past a null-valued record and past an
    undecodable-JSON record, after ``async for`` had already advanced the
    consumer's POSITION past them. Neither was handed to the handler, neither
    was quarantined, and the trailing bare ``commit()`` -- which commits the
    position, not a set of completed offsets -- committed them away. Measured
    below on the real pinned code: valid @0 / undecodable @1 / valid @2 gives
    ``total_processed=2``, ``quarantined=0``, ``commits=[3]``, and a normal
    bounded-batch summary. Offset 1 silently lost.
  * ``_quarantine`` caught a failed quarantine publish and returned a
    ``FAILED`` result rather than raising, after which ``run()`` counted the
    record and committed it -- the offset advanced past a record whose
    quarantine never became durable.
  * Both commits were bare ``commit()`` calls, so ANY early exit (deadline,
    cancellation) advanced past the record that was in flight.

THE ONE ADMISSIBLE SHAPE. The generator does not publish. Injecting the
quarantine producer into ``DLQConsumer`` and publishing from the yield boundary
puts a network round-trip inside ``__anext__``, which is awaited under
``asyncio.wait_for(..., timeout=remaining)``: a deadline landing mid-publish
CANCELS the publish, ``run()`` catches ``TimeoutError`` as the ordinary idle
topic case, breaks, and commits -- the malformed record durably lost while the
run reports a clean bounded batch. So ``consume_messages`` yields a typed
``ModelUnparseableDlqRecord`` and every durable write happens on ``run()``'s
own frame, outside the guarded acquisition.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest

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

_LIVE_POISONED_TOPIC = "onex.evt.omniclaude.tool-executed.v1"  # onex-topic-allow: quotes the measured live record
_LIVE_DLQ_TOPIC = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: the live DLQ this node drains


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": _LIVE_DLQ_TOPIC,
        "max_records_per_run": 50,
        "commit_every_n_records": 1000,
        "max_run_duration_seconds": 5.0,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _valid_dlq_payload() -> bytes:
    return json.dumps(
        {
            "original_topic": _LIVE_POISONED_TOPIC,
            "original_message": {"key": "k", "value": '{"hello": "world"}'},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": 0,
        }
    ).encode("utf-8")


def _absent_value_dlq_payload() -> bytes:
    """Decodes as JSON; its ``original_message`` has no ``value`` key at all."""
    return json.dumps(
        {
            "original_topic": _LIVE_POISONED_TOPIC,
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
    """Mirrors aiokafka: ``__anext__`` advances the consumer POSITION past a
    record before the caller sees it, and a bare ``commit()`` commits that
    position -- which is exactly why a skipped record is committed away."""

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
    """The REAL ``DLQConsumer`` with only the aiokafka client stubbed."""
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
    """Records every quarantine publish. ``fail`` makes the publish raise, and
    ``block`` makes it hang so a cancellation can land mid-publish."""

    def __init__(self, *, fail: bool = False, block: bool = False) -> None:
        self._started = True
        self.fail = fail
        self.block = block
        self.calls: list[tuple[str, object, str]] = []
        self.entered = asyncio.Event()

    async def start(self) -> None:  # pragma: no cover - not reached
        return None

    async def stop(self) -> None:  # pragma: no cover - not reached
        return None

    async def quarantine_message(
        self, message: object, reason: str, _cid: object
    ) -> object:
        self.calls.append(("parsed", message, reason))
        if self.fail:
            raise RuntimeError("quarantine publish failed")
        return object()

    async def quarantine_unparseable_record(
        self, record: ModelUnparseableDlqRecord, _cid: object
    ) -> object:
        self.calls.append(("unparseable", record, record.reason))
        self.entered.set()
        if self.block:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("quarantine publish failed")
        return object()


def _handler(
    consumer: DLQConsumer,
    producer: _NoopReplayProducer,
    quarantine: _RecordingQuarantineProducer,
) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumers={consumer.config.dlq_topic: consumer},
        producer=producer,  # type: ignore[arg-type]
        quarantine_producer=quarantine,  # type: ignore[arg-type]
        tracking=None,
    )


# --------------------------------------------------------------------------
# RED test (6) -- the refusal reaches quarantine instead of aborting the drain
# --------------------------------------------------------------------------


async def test_6_a_parse_refusal_is_quarantined_and_the_drain_continues() -> None:
    """RED on origin/dev: ``run()`` raises ``ValueError`` out of ``anext``,
    the third record is never consumed and nothing is quarantined."""
    config = _config()
    poison = _absent_value_dlq_payload()
    consumer, fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(_valid_dlq_payload(), 0),
            _FakeConsumerRecord(poison, 1),
            _FakeConsumerRecord(_valid_dlq_payload(), 2),
        ],
    )
    producer = _NoopReplayProducer()
    quarantine = _RecordingQuarantineProducer()

    result = await _handler(consumer, producer, quarantine).run()

    assert result.total_processed == 3, result
    assert len(producer.replayed) == 2, "both valid records must still replay"
    unparseable = [c for c in quarantine.calls if c[0] == "unparseable"]
    assert len(unparseable) == 1, quarantine.calls
    record = unparseable[0][1]
    assert isinstance(record, ModelUnparseableDlqRecord)
    assert record.dlq_offset == 1
    assert record.raw_value == poison, (
        "the quarantine record must carry the record's RAW BYTES -- they are "
        "the only thing that makes it reclassifiable later"
    )
    assert "value" in record.reason, record.reason
    assert fake.commits, "the batch must commit"
    assert _max_committed(fake.commits) >= 3, fake.commits


# --------------------------------------------------------------------------
# RED test (7) -- a failed quarantine never advances the offset
# --------------------------------------------------------------------------


async def test_7_a_failed_quarantine_publish_withholds_the_offset() -> None:
    """RED on origin/dev: ``_quarantine`` swallows the failure into a FAILED
    result, ``run()`` counts the record and the trailing commit advances past
    it -- the silent drop, reached through the quarantine path."""
    config = _config()
    poison = json.dumps(
        {
            "original_topic": _LIVE_POISONED_TOPIC,
            "original_message": {"key": "k", "value": ""},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": 0,
        }
    ).encode("utf-8")
    consumer, fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(_valid_dlq_payload(), 0),
            _FakeConsumerRecord(poison, 1),
        ],
    )
    producer = _NoopReplayProducer()
    quarantine = _RecordingQuarantineProducer(fail=True)

    result = await _handler(consumer, producer, quarantine).run()

    assert result.failed == 1, result
    assert _max_committed(fake.commits) <= 1, (
        "the offset advanced past a record whose quarantine publish failed: "
        f"commits={fake.commits}"
    )


async def test_7b_a_successful_quarantine_publish_does_advance_the_offset() -> None:
    """The positive control for (7). An implementation that withholds the
    offset unconditionally passes (7) and is a permanent stall."""
    config = _config()
    poison = json.dumps(
        {
            "original_topic": _LIVE_POISONED_TOPIC,
            "original_message": {"key": "k", "value": ""},
            "correlation_id": str(uuid4()),
            "error_type": "InfraConnectionError",
            "retry_count": 0,
        }
    ).encode("utf-8")
    consumer, fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(_valid_dlq_payload(), 0),
            _FakeConsumerRecord(poison, 1),
        ],
    )
    quarantine = _RecordingQuarantineProducer(fail=False)
    result = await _handler(consumer, _NoopReplayProducer(), quarantine).run()

    assert result.quarantined == 1, result
    assert _max_committed(fake.commits) >= 2, fake.commits


# --------------------------------------------------------------------------
# RED test (8) -- no durable write awaited inside the guarded acquisition
# --------------------------------------------------------------------------


def test_8a_the_consumer_holds_no_quarantine_collaborator() -> None:
    """A regression fence, GREEN on ``origin/dev`` and required to stay green:
    it is the structural form of "the generator does not publish". Exempt from
    the RED preamble by construction -- ``origin/dev`` has no in-generator
    publish to fail on."""
    import inspect

    params = list(inspect.signature(DLQConsumer.__init__).parameters)
    assert params == ["self", "config"], params
    consumer = DLQConsumer(_config())
    attrs = vars(consumer)
    assert not any("quarantine" in name for name in attrs), attrs


async def test_8b_the_generator_yields_the_refusal_without_publishing() -> None:
    """RED on origin/dev: the typed value does not exist, so ``anext`` skips
    the record entirely and raises ``StopAsyncIteration``.

    Drives ``consume_messages`` DIRECTLY -- no ``run()`` -- with an
    instrumented quarantine producer that the generator has no way to reach.
    Zero calls is the assertion: every durable write happens on ``run()``'s own
    frame, outside ``asyncio.wait_for(anext(...), timeout=remaining)``.
    """
    config = _config()
    consumer, _fake = _wired_consumer(
        config, [_FakeConsumerRecord(_absent_value_dlq_payload(), 1)]
    )
    quarantine = _RecordingQuarantineProducer()

    messages = consumer.consume_messages()
    try:
        yielded = await anext(messages)
    finally:
        await messages.aclose()

    assert isinstance(yielded, ModelUnparseableDlqRecord), type(yielded).__name__
    assert quarantine.calls == [], quarantine.calls


# --------------------------------------------------------------------------
# RED test (9) -- the commit covers only the records whose handling completed
# --------------------------------------------------------------------------


def _max_committed(commits: list[Any]) -> int:
    """Highest committed offset boundary across bare-position and offset-map
    commits, so the assertion reads the same on both shapes."""
    highest = 0
    for entry in commits:
        if isinstance(entry, int):
            highest = max(highest, entry)
        elif isinstance(entry, dict):
            for value in entry.values():
                highest = max(highest, int(getattr(value, "offset", value)))
    return highest


async def test_9_an_undecodable_record_is_quarantined_not_committed_away() -> None:
    """RED on origin/dev, reproduced against the real pinned generator and
    ``run()``: valid @0 / undecodable-JSON @1 / valid @2 returns
    ``total_processed=2``, ``quarantined=0``, ``commits=[3]`` -- offset 1
    ``continue``d past inside the generator after the position had already
    advanced, then committed away by the trailing bare ``commit()``, and the
    run reported a normal bounded batch."""
    config = _config()
    consumer, fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(_valid_dlq_payload(), 0),
            _FakeConsumerRecord(b"{not json at all", 1),
            _FakeConsumerRecord(_valid_dlq_payload(), 2),
        ],
    )
    quarantine = _RecordingQuarantineProducer()
    result = await _handler(consumer, _NoopReplayProducer(), quarantine).run()

    assert result.total_processed == 3, result
    assert result.quarantined == 1, result
    unparseable = [c for c in quarantine.calls if c[0] == "unparseable"]
    assert len(unparseable) == 1, quarantine.calls
    assert unparseable[0][1].raw_value == b"{not json at all"
    assert _max_committed(fake.commits) >= 3, fake.commits


async def test_9b_a_null_valued_record_is_quarantined_not_committed_away() -> None:
    """The generator's OTHER bare ``continue``, same class, same silent drop."""
    config = _config()
    consumer, fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(None, 0),
            _FakeConsumerRecord(_valid_dlq_payload(), 1),
        ],
    )
    quarantine = _RecordingQuarantineProducer()
    result = await _handler(consumer, _NoopReplayProducer(), quarantine).run()

    assert result.total_processed == 2, result
    unparseable = [c for c in quarantine.calls if c[0] == "unparseable"]
    assert len(unparseable) == 1, quarantine.calls
    assert unparseable[0][1].raw_value is None
    assert _max_committed(fake.commits) >= 2, fake.commits


async def test_9c_a_cancellation_mid_quarantine_does_not_commit_that_record() -> None:
    """A lane shutdown while the quarantine publish for offset 1 is in flight
    must leave offset 1 uncommitted, so the record is redelivered."""
    config = _config(commit_every_n_records=1, max_run_duration_seconds=30.0)
    consumer, fake = _wired_consumer(
        config,
        [
            _FakeConsumerRecord(_valid_dlq_payload(), 0),
            _FakeConsumerRecord(b"{not json at all", 1),
            _FakeConsumerRecord(_valid_dlq_payload(), 2),
        ],
    )
    quarantine = _RecordingQuarantineProducer(block=True)
    handler = _handler(consumer, _NoopReplayProducer(), quarantine)

    task = asyncio.create_task(handler.run())
    await asyncio.wait_for(quarantine.entered.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _max_committed(fake.commits) <= 1, (
        "a commit covered the record whose quarantine publish was cancelled: "
        f"commits={fake.commits}"
    )

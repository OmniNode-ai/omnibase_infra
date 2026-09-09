# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18084 — the replay producer never publishes onto a dead-letter topic.

THE FOURTH FIXED POINT. #3365 closed three of the four surfaces that fed the
.201 dev-lane DLQ loop: the unguarded ``HANDLER_ERROR`` leg, the
``get_dlq_topic_for_original`` resolver, and the shared ``DLQConsumer``
lifecycle. It did not touch the replay producer itself. ``replay_message``
sends to ``message.original_topic`` UNCONDITIONALLY, and ``is_dlq_topic``
appears nowhere in ``engine_dlq_replay.py`` — so a DLQ record whose
``original_topic`` IS ``onex.dlq.omnibase-infra.events.v1`` is published
straight back onto the topic it was consumed from.

WHAT WAS MEASURED (dev lane ``omnibase-infra``, .201, 2026-09-09, read-only).
After #3365 deployed at 19:28Z the loop fell from 182.9 records/s to 34.4/s but
did NOT close, because the retained backlog is ~1.67M records that are already
NESTED dead letters. One record read at offset 21685100 (14,939 bytes) unwraps
like this, layer by layer::

    0..6  original_topic = onex.dlq.omnibase-infra.events.v1   error_type = HandlerDispatchFailureError
    7     original_topic = onex.evt.omniclaude.prompt-submitted.v1
          error_type = JSONDecodeError   retry_count = 4   original_message.value = ""

Seven envelope layers wrapping one real record. Each replay pass strips exactly
ONE layer and writes the inner envelope back onto the same DLQ topic, so the
population is self-terminating per record and still worth order 10M further
writes at 34/s against a mount shared with the prod, stability-test and judge
lanes.

WHY UNWRAPPING IS THE FIX AND NOT MERELY A REFUSAL. Refusing the fixed point
alone would quarantine ~1.67M records at their OUTER size (~25 KB each). Reading
the record instead — down to the innermost envelope whose ``original_topic`` is a
real source topic — costs one JSON parse per layer and hands ``should_replay``
the record the DLQ was actually built to preserve. On the live population the
innermost body is EMPTY, so the OMN-17896 empty-body clause refuses it and it
takes the durable quarantine exit at ~200 bytes rather than 25 KB. Where an
innermost body IS replayable, it is replayed ONCE to its real topic instead of
being peeled one layer per pass.

Three properties are asserted here, each with a positive control:

(a) ``replay_message`` refuses a DLQ destination outright, and ``should_replay``
    refuses the same record as an eligibility matter, so a caller that never
    unwraps still cannot feed the loop.
(b) A nested record is unwrapped to its innermost real original topic and
    replayed EXACTLY ONCE there — not layer by layer.
(c) The existing non-nested path is byte-for-byte unchanged.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.errors import DlqTopicFixedPointError
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    MAX_DLQ_UNWRAP_DEPTH,
    DLQProducer,
    ModelDlqReplayEngineConfig,
    should_replay,
    unwrap_nested_dlq_record,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

# onex-topic-allow: the live DLQ topic this node drains, quoted from the
# measured .201 dev-lane record at offset 21685100.
_DLQ_TOPIC = "onex.dlq.omnibase-infra.events.v1"
# onex-topic-allow: the innermost original topic of that same measured record.
_REAL_TOPIC = "onex.evt.omniclaude.prompt-submitted.v1"
# onex-topic-allow: a second real source topic, used as the replayable-innermost
# positive control so the unwrap assertions are not single-topic artifacts.
_OTHER_REAL_TOPIC = "onex.evt.omniclaude.tool-executed.v1"
# onex-topic-allow: the durable quarantine sink (OMN-12619).
_QUARANTINE_TOPIC = "onex.dlq.omnibase-infra.quarantine.v1"


def _config(**overrides: Any) -> ModelDlqReplayEngineConfig:
    kwargs: dict[str, Any] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": _DLQ_TOPIC,
        "quarantine_topic": _QUARANTINE_TOPIC,
    }
    kwargs.update(overrides)
    return ModelDlqReplayEngineConfig(**kwargs)


def _dlq_message(
    *,
    original_topic: str,
    original_value: str,
    retry_count: int = 0,
    error_type: str = "HandlerDispatchFailureError",
    dlq_offset: int = 21685100,
    dlq_partition: int = 0,
    correlation_id: UUID | None = None,
) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=original_topic,
        original_value=original_value,
        correlation_id=correlation_id or uuid4(),
        retry_count=retry_count,
        error_type=error_type,
        dlq_offset=dlq_offset,
        dlq_partition=dlq_partition,
    )


def _dlq_payload(
    *,
    original_topic: str,
    value: str,
    retry_count: int = 0,
    error_type: str = "HandlerDispatchFailureError",
) -> dict[str, object]:
    """One DLQ envelope in the exact shape ``mixin_kafka_dlq`` publishes.

    Field names and nesting are read off ``MixinKafkaDlq`` (``original_topic`` /
    ``original_message.{key,value,offset,partition}`` / ``failure_reason`` /
    ``failure_timestamp`` / ``correlation_id`` / ``retry_count`` /
    ``error_type``) and cross-checked against the live 14,939-byte record, so a
    fixture drift would be a real drift rather than a test-only mismatch.
    """
    return {
        "original_topic": original_topic,
        "original_message": {
            "key": None,
            "value": value,
            "offset": 6574,
            "partition": 0,
        },
        "failure_reason": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        "failure_timestamp": "2026-09-09T18:22:41.104312+00:00",
        "correlation_id": str(uuid4()),
        "retry_count": retry_count,
        "error_type": error_type,
        "failure_type": "handler_exception",
    }


def _nest(
    *,
    layers: int,
    innermost_topic: str,
    innermost_value: str,
    innermost_retry_count: int = 4,
    innermost_error_type: str = "JSONDecodeError",
) -> str:
    """Wrap ``innermost_value`` in ``layers`` dead-letter envelopes.

    ``layers=7`` with the defaults reproduces the measured live record: seven
    envelopes whose ``original_topic`` is the DLQ topic itself, wrapping one
    record on ``onex.evt.omniclaude.prompt-submitted.v1`` that carries
    ``retry_count=4`` and ``error_type=JSONDecodeError``.

    Both innermost fields are parameters because the eligibility clauses are
    evaluated against the UNWRAPPED record, so the fixture has to be able to
    express an innermost record that is genuinely replayable. ``JSONDecodeError``
    is in ``EnumNonRetryableErrorCategory``, which is why the default is the
    right shape for the live population and the wrong one for a replay control.
    """
    value = json.dumps(
        _dlq_payload(
            original_topic=innermost_topic,
            value=innermost_value,
            retry_count=innermost_retry_count,
            error_type=innermost_error_type,
        )
    )
    for _ in range(layers - 1):
        value = json.dumps(_dlq_payload(original_topic=_DLQ_TOPIC, value=value))
    return value


def _replayable_body() -> str:
    """A body no clause in ``should_replay`` refuses."""
    return json.dumps(
        {
            "payload_type": "ModelToolExecuted",
            "event_type": "omniclaude.tool-executed",
            "correlation_id": str(uuid4()),
            "payload": {"tool": "Bash", "exit_code": 0},
        }
    )


class _FakeKafkaProducer:
    """Captures every ``send_and_wait`` so a refusal proves ZERO publishes."""

    def __init__(self) -> None:
        self.sends: list[tuple[str, bytes | None]] = []

    async def send_and_wait(
        self,
        topic: str,
        *,
        value: bytes | None = None,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object:
        self.sends.append((topic, value))
        return object()


def _started_producer() -> tuple[DLQProducer, _FakeKafkaProducer]:
    producer = DLQProducer(_config())
    fake = _FakeKafkaProducer()
    producer._producer = fake  # type: ignore[assignment]
    producer._started = True
    return producer, fake


class _FakeQuarantineProducer:
    def __init__(self) -> None:
        self.quarantined: list[tuple[ModelDlqMessage, str]] = []

    async def quarantine_message(
        self,
        message: ModelDlqMessage,
        reason: str,
        quarantine_correlation_id: UUID,
    ) -> object:
        self.quarantined.append((message, reason))
        return object()


class _FakeConsumer:
    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config


def _handler(
    config: ModelDlqReplayEngineConfig | None = None,
) -> tuple[HandlerDlqReplay, _FakeKafkaProducer, _FakeQuarantineProducer]:
    cfg = config or _config()
    producer, fake_kafka = _started_producer()
    producer.config = cfg  # type: ignore[misc]
    quarantine = _FakeQuarantineProducer()
    handler = HandlerDlqReplay(
        consumer=_FakeConsumer(cfg),  # type: ignore[arg-type]
        producer=producer,
        quarantine_producer=quarantine,  # type: ignore[arg-type]
        tracking=None,
    )
    return handler, fake_kafka, quarantine


# ---------------------------------------------------------------------------
# (a) The producer refuses a dead-letter destination
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replay_message_refuses_to_publish_onto_a_dlq_topic() -> None:
    """RED before the fix: this call published onto the topic it read from."""
    producer, fake = _started_producer()
    message = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(layers=2, innermost_topic=_REAL_TOPIC, innermost_value=""),
    )

    with pytest.raises(DlqTopicFixedPointError) as excinfo:
        await producer.replay_message(message, uuid4())

    assert _DLQ_TOPIC in str(excinfo.value)
    assert fake.sends == [], (
        "replay_message published onto a dead-letter topic; on the .201 dev "
        "lane that is the write that kept the loop alive at 34.4 records/s "
        "after #3365 deployed"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replay_message_still_publishes_to_a_real_original_topic() -> None:
    """POSITIVE CONTROL: the refusal is destination-typed, not a blanket stop."""
    producer, fake = _started_producer()
    message = _dlq_message(
        original_topic=_REAL_TOPIC, original_value=_replayable_body()
    )

    await producer.replay_message(message, uuid4())

    assert [topic for topic, _ in fake.sends] == [_REAL_TOPIC]


@pytest.mark.unit
def test_should_replay_refuses_a_record_whose_original_topic_is_a_dlq_topic() -> None:
    """The eligibility predicate refuses it too, so a non-unwrapping caller
    (``scripts/dlq_replay.py``) takes the durable quarantine exit rather than
    reaching the producer's last-resort raise, which would block the partition.
    """
    eligible, reason = should_replay(
        _dlq_message(
            original_topic=_DLQ_TOPIC,
            original_value=_nest(
                layers=2, innermost_topic=_REAL_TOPIC, innermost_value=""
            ),
        ),
        _config(),
    )
    assert eligible is False
    assert "dead-letter" in reason.lower() or "dlq" in reason.lower(), reason


@pytest.mark.unit
def test_should_replay_still_accepts_a_real_original_topic() -> None:
    """POSITIVE CONTROL for the clause above."""
    eligible, _ = should_replay(
        _dlq_message(original_topic=_REAL_TOPIC, original_value=_replayable_body()),
        _config(),
    )
    assert eligible is True


# ---------------------------------------------------------------------------
# (b) A nested record is unwrapped to its innermost real original topic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_measured_live_record_unwraps_seven_layers_to_its_real_topic() -> None:
    """Reproduces the record read at offset 21685100 on the dev lane."""
    outer = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(layers=7, innermost_topic=_REAL_TOPIC, innermost_value=""),
    )

    unwrapped, depth = unwrap_nested_dlq_record(outer)

    assert depth == 7
    assert unwrapped.original_topic == _REAL_TOPIC
    assert unwrapped.original_value == ""
    assert unwrapped.retry_count == 4
    assert unwrapped.error_type == "JSONDecodeError"


@pytest.mark.unit
def test_unwrapping_preserves_the_outer_dlq_coordinates() -> None:
    """The record physically lives at the OUTER offset.

    ``_mark_offset`` commits ``message.dlq_offset + 1`` on the drained topic, so
    an unwrapped message carrying the INNER envelope's offset (6574) would
    rewind the replay group by ~21.6M records.
    """
    outer = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(layers=4, innermost_topic=_REAL_TOPIC, innermost_value=""),
        dlq_offset=21685100,
        dlq_partition=0,
    )

    unwrapped, _ = unwrap_nested_dlq_record(outer)

    assert unwrapped.dlq_offset == 21685100
    assert unwrapped.dlq_partition == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_nested_record_is_replayed_once_to_the_real_topic_not_layer_by_layer() -> (
    None
):
    """The whole point of (b): ONE publish, to the real topic, per record."""
    handler, fake_kafka, quarantine = _handler()
    message = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(
            layers=6,
            innermost_topic=_OTHER_REAL_TOPIC,
            innermost_value=_replayable_body(),
            innermost_retry_count=1,
            innermost_error_type="HandlerDispatchFailureError",
        ),
    )

    result = await handler._process_message(message)

    assert result.status == EnumReplayStatus.COMPLETED
    assert [topic for topic, _ in fake_kafka.sends] == [_OTHER_REAL_TOPIC], (
        "a nested record must reach its innermost real topic in one pass; "
        "publishing one layer at a time is what produced order 10M writes"
    )
    assert quarantine.quarantined == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_live_nested_population_is_quarantined_and_never_republished() -> (
    None
):
    """The measured population unwraps to an EMPTY innermost body.

    OMN-17896's empty-body clause then refuses it, so it takes the durable
    quarantine exit — and, critically, the quarantine record carries the
    ~200-byte INNERMOST payload rather than the ~25 KB outer nest.
    """
    handler, fake_kafka, quarantine = _handler()
    message = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(layers=7, innermost_topic=_REAL_TOPIC, innermost_value=""),
    )

    result = await handler._process_message(message)

    assert result.status == EnumReplayStatus.QUARANTINED
    assert fake_kafka.sends == [], "zero writes back onto any dead-letter topic"
    assert len(quarantine.quarantined) == 1
    quarantined_message, reason = quarantine.quarantined[0]
    assert quarantined_message.original_topic == _REAL_TOPIC
    assert "unwrapped" in reason.lower(), reason
    assert "empty" in reason.lower(), reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eligibility_is_decided_on_the_innermost_record_not_the_envelope() -> (
    None
):
    """Unwrapping is not cosmetic: every existing clause now sees the real record.

    The outer envelopes all carry ``error_type=HandlerDispatchFailureError``,
    which is retryable. The innermost carries ``JSONDecodeError``, which is in
    ``EnumNonRetryableErrorCategory``. The record must be refused on the
    INNERMOST classification and the reason must name it, otherwise the unwrap
    has merely changed the destination and kept the wrong verdict.
    """
    handler, fake_kafka, quarantine = _handler()
    message = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(
            layers=5,
            innermost_topic=_OTHER_REAL_TOPIC,
            innermost_value=_replayable_body(),
            innermost_retry_count=0,
            innermost_error_type="JSONDecodeError",
        ),
        error_type="HandlerDispatchFailureError",
    )

    result = await handler._process_message(message)

    assert result.status == EnumReplayStatus.QUARANTINED
    assert fake_kafka.sends == []
    _, reason = quarantine.quarantined[0]
    assert "JSONDecodeError" in reason, reason
    assert "non-retryable" in reason.lower(), reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_record_whose_nest_cannot_be_read_is_quarantined_not_replayed() -> None:
    """An unreadable inner layer terminalises; it never falls through to a send."""
    handler, fake_kafka, quarantine = _handler()
    message = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value="this is not a dead-letter envelope at all",
    )

    result = await handler._process_message(message)

    assert result.status == EnumReplayStatus.QUARANTINED
    assert fake_kafka.sends == []
    _, reason = quarantine.quarantined[0]
    assert _DLQ_TOPIC in reason


@pytest.mark.unit
def test_unwrapping_refuses_beyond_the_depth_cap() -> None:
    """A cap, not a ``while True``: a cyclic or adversarial nest terminates."""
    outer = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(
            layers=MAX_DLQ_UNWRAP_DEPTH + 2,
            innermost_topic=_REAL_TOPIC,
            innermost_value="",
        ),
    )

    with pytest.raises(DlqTopicFixedPointError) as excinfo:
        unwrap_nested_dlq_record(outer)

    assert str(MAX_DLQ_UNWRAP_DEPTH) in str(excinfo.value)


@pytest.mark.unit
def test_unwrapping_reaches_the_real_topic_at_exactly_the_depth_cap() -> None:
    """POSITIVE CONTROL for the cap: the boundary case succeeds."""
    outer = _dlq_message(
        original_topic=_DLQ_TOPIC,
        original_value=_nest(
            layers=MAX_DLQ_UNWRAP_DEPTH,
            innermost_topic=_REAL_TOPIC,
            innermost_value="",
        ),
    )

    unwrapped, depth = unwrap_nested_dlq_record(outer)

    assert depth == MAX_DLQ_UNWRAP_DEPTH
    assert unwrapped.original_topic == _REAL_TOPIC


# ---------------------------------------------------------------------------
# (c) The existing non-nested path is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_non_nested_record_is_returned_unchanged_and_unparsed() -> None:
    """Identity, not a rebuild: the non-nested path must not touch the record.

    ``is`` rather than ``==`` because a rebuild would silently re-run
    ``from_kafka_message`` on every ordinary DLQ record in the fleet.
    """
    message = _dlq_message(
        original_topic=_REAL_TOPIC, original_value="not json, and never parsed"
    )

    unwrapped, depth = unwrap_nested_dlq_record(message)

    assert unwrapped is message
    assert depth == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_ordinary_replay_path_still_replays_once_to_the_original_topic() -> (
    None
):
    """POSITIVE CONTROL for (c) at the handler level."""
    handler, fake_kafka, quarantine = _handler()
    message = _dlq_message(
        original_topic=_REAL_TOPIC, original_value=_replayable_body()
    )

    result = await handler._process_message(message)

    assert result.status == EnumReplayStatus.COMPLETED
    assert [topic for topic, _ in fake_kafka.sends] == [_REAL_TOPIC]
    assert quarantine.quarantined == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_ordinary_ineligible_record_is_quarantined_with_an_unprefixed_reason() -> (
    None
):
    """The unwrap note appears only when something was actually unwrapped."""
    handler, _, quarantine = _handler()
    message = _dlq_message(original_topic=_REAL_TOPIC, original_value="", retry_count=0)

    result = await handler._process_message(message)

    assert result.status == EnumReplayStatus.QUARANTINED
    _, reason = quarantine.quarantined[0]
    assert "unwrapped" not in reason.lower(), reason
    assert "empty" in reason.lower(), reason

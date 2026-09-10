# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17896 -- the DLQ replay effect manufactured a zero-byte record and then
treated the guaranteed decode failure as retryable.

WHAT WAS MEASURED (dev lane compose project ``omnibase-infra``, 2026-09-07,
read-only). Every one of the last 300 records on
``onex.evt.omniclaude.tool-executed.v1`` carried
``x-replayed-by=node_dlq_replay_effect`` with ``x-original-dlq-offset`` and an
incrementing ``x-replay-count``; 200 of 200 sampled had ``VALSZ=0`` AND
``KEYSZ=0``. Produce rate over 30 s: 104 records, ~3.5/s sustained, re-measured
at ~4.3/s later the same day. The consumer side of that loop logged 5,535
``JSONDecodeError: Expecting value: line 1 column 1 (char 0)`` lines -- the
empty-body signature -- in one five-minute window, out of 27,109 total lines
(positive control in the same window: 14,619 lines matching a token known to be
present; negative control: 0).

TWO SILENT EMPTY DEFAULTS IN SERIES MANUFACTURED THAT RECORD.

  1. ``MixinKafkaDlq._publish_raw_to_dlq`` read the failed record's body as
     ``getattr(raw_msg, "value", b"")``, so a message object whose value could
     not be read produced a durable DLQ record asserting ``"value": ""`` --
     an empty body it never had. (Asserted in
     ``tests/unit/event_bus/test_omn17896_raw_dlq_unreadable_value.py``.)
  2. ``ModelDlqMessage.from_kafka_message`` read it back as
     ``str(original_message.get("value", ""))``, converting a missing key into
     a publishable empty string.

``DLQProducer.replay_message`` then published ``original_value.encode(...)``
-- zero bytes -- to the original topic. The consumer could not decode it, and
``JSONDecodeError`` was absent from ``EnumNonRetryableErrorCategory``, so the
record stayed replay-eligible and the cycle repeated to the cap. The cap bounds
one CHAIN; it does not stop new chains being minted, which is why the topic
sustained ~4 records/s of guaranteed-undecodable traffic.

Every test in this module fails on ``origin/dev`` (8d4e62f78).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQ_UNREADABLE_VALUE_MARKER,
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqRecordUnparseableError,
)

pytestmark = pytest.mark.unit

_LIVE_POISONED_TOPIC = "onex.evt.omniclaude.tool-executed.v1"  # onex-topic-allow: quotes the measured live record
_LIVE_DLQ_TOPIC = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: the live DLQ this node drains


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": _LIVE_DLQ_TOPIC,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _message(original_value: str) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_LIVE_POISONED_TOPIC,
        original_key=None,
        original_value=original_value,
        correlation_id=uuid4(),
        retry_count=0,
        error_type="InfraConnectionError",  # retryable -> eligible but for the body
        dlq_offset=9588531,
        dlq_partition=0,
    )


# --------------------------------------------------------------------------
# RED test (1) -- the eligibility predicate has no empty-body rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("empty_body", ["", "   ", "\n\t "])
def test_1_an_empty_original_body_is_not_eligible_for_replay(empty_body: str) -> None:
    """RED on origin/dev: returns ``(True, "Eligible for replay")``.

    A body that is empty (or whitespace only) cannot be JSON-decoded by ANY
    consumer of the original topic, so the replay is guaranteed to fail --
    which is the whole 4-records/s loop.
    """
    eligible, reason = should_replay(_message(empty_body), _config())
    assert eligible is False, (
        "the replay engine published a zero-byte record onto "
        f"{_LIVE_POISONED_TOPIC}; 200 of 200 sampled live records were "
        "zero-byte and replay-stamped"
    )
    assert "empty" in reason.lower(), reason


def test_1b_an_unreadable_original_body_is_not_eligible_for_replay() -> None:
    """The unreadable marker is not a body either -- it is a statement that
    the body could not be read. Publishing those literal bytes onto the
    original topic would be a second manufactured record."""
    eligible, reason = should_replay(_message(DLQ_UNREADABLE_VALUE_MARKER), _config())
    assert eligible is False
    assert "unreadable" in reason.lower(), reason


def test_1c_a_real_body_is_still_eligible() -> None:
    """The anti-over-refusal control. A guard that refuses EVERY replay passes
    test (1) trivially and is a regression, not a repair."""
    eligible, reason = should_replay(_message('{"hello": "world"}'), _config())
    assert eligible is True, reason


# --------------------------------------------------------------------------
# RED test (2) -- the second silent empty default
# --------------------------------------------------------------------------


def test_2_from_kafka_message_refuses_an_absent_value_key() -> None:
    """RED on origin/dev: succeeds with ``original_value == ""``."""
    with pytest.raises(DlqRecordUnparseableError) as excinfo:
        ModelDlqMessage.from_kafka_message(
            payload={
                "original_topic": _LIVE_POISONED_TOPIC,
                "original_message": {"key": "k"},
                "correlation_id": str(uuid4()),
            },
            dlq_offset=9588531,
            dlq_partition=0,
        )
    assert "value" in str(excinfo.value), str(excinfo.value)


def test_2b_from_kafka_message_refuses_a_null_value() -> None:
    """``None`` is not a body either; ``str(None)`` would have published the
    four literal bytes ``None`` onto the original topic."""
    with pytest.raises(DlqRecordUnparseableError):
        ModelDlqMessage.from_kafka_message(
            payload={
                "original_topic": _LIVE_POISONED_TOPIC,
                "original_message": {"key": "k", "value": None},
                "correlation_id": str(uuid4()),
            },
            dlq_offset=9588531,
            dlq_partition=0,
        )


def test_2c_from_kafka_message_still_parses_a_well_formed_record() -> None:
    """Positive control for tests (2) and (2b): the refusal is not universal."""
    parsed = ModelDlqMessage.from_kafka_message(
        payload={
            "original_topic": _LIVE_POISONED_TOPIC,
            "original_message": {"key": "k", "value": '{"hello": "world"}'},
            "correlation_id": str(uuid4()),
        },
        dlq_offset=9588531,
        dlq_partition=0,
    )
    assert parsed.original_value == '{"hello": "world"}'


def test_2d_an_empty_value_string_parses_and_is_refused_downstream() -> None:
    """An EMPTY value that is genuinely present is a different fact from an
    ABSENT one, and the two must not be conflated: the record parses (the
    field was there) and is then refused by ``should_replay`` (test 1), so it
    reaches the durable quarantine path rather than aborting the parse."""
    parsed = ModelDlqMessage.from_kafka_message(
        payload={
            "original_topic": _LIVE_POISONED_TOPIC,
            "original_message": {"key": "k", "value": ""},
            "correlation_id": str(uuid4()),
        },
        dlq_offset=9588531,
        dlq_partition=0,
    )
    assert parsed.original_value == ""
    eligible, _ = should_replay(parsed, _config())
    assert eligible is False


# --------------------------------------------------------------------------
# RED test (4) -- a body that can never decode is treated as retryable
# --------------------------------------------------------------------------


def test_4_json_decode_error_is_non_retryable() -> None:
    """RED on origin/dev: the set holds exactly five members and this is not
    one of them, so a structurally undecodable body was replayed to the cap
    (five rounds) instead of exiting on its first pass."""
    assert "JSONDecodeError" in EnumNonRetryableErrorCategory.get_all_values()
    assert EnumNonRetryableErrorCategory.is_non_retryable("JSONDecodeError")


def test_4b_a_dlq_record_classified_json_decode_error_is_not_replayed() -> None:
    """The enum membership matters only because ``should_replay`` reads it."""
    message = ModelDlqMessage(
        original_topic=_LIVE_POISONED_TOPIC,
        original_value='{"still": "decodable here"}',
        correlation_id=uuid4(),
        retry_count=0,
        error_type="JSONDecodeError",
        dlq_offset=9588531,
        dlq_partition=0,
    )
    eligible, reason = should_replay(message, _config())
    assert eligible is False, reason
    assert "JSONDecodeError" in reason


# --------------------------------------------------------------------------
# RED test (5) -- end to end at the node
# --------------------------------------------------------------------------


class _RecordingProducer:
    def __init__(self) -> None:
        self._started = True
        self.replayed: list[ModelDlqMessage] = []

    async def start(self) -> None:  # pragma: no cover - not reached
        self._started = True

    async def stop(self) -> None:  # pragma: no cover - not reached
        self._started = False

    async def replay_message(self, message: ModelDlqMessage, _cid: object) -> None:
        self.replayed.append(message)


class _RecordingQuarantineProducer:
    def __init__(self) -> None:
        self._started = True
        self.quarantined: list[tuple[ModelDlqMessage, str]] = []

    async def start(self) -> None:  # pragma: no cover - not reached
        self._started = True

    async def stop(self) -> None:  # pragma: no cover - not reached
        self._started = False

    async def quarantine_message(
        self, message: ModelDlqMessage, reason: str, _cid: object
    ) -> object:
        self.quarantined.append((message, reason))
        return object()


async def test_5_an_empty_bodied_dlq_record_is_quarantined_not_republished() -> None:
    """RED on origin/dev: ``replayed`` holds one message and a zero-byte
    record goes onto the original topic."""
    config = _config()
    producer = _RecordingProducer()
    quarantine = _RecordingQuarantineProducer()

    class _Consumer:
        config = None  # set below
        _started = True

        async def start(self) -> None:  # pragma: no cover - not reached
            return None

        async def stop(self) -> None:  # pragma: no cover - not reached
            return None

    consumer = _Consumer()
    consumer.config = config  # type: ignore[assignment]

    handler = HandlerDlqReplay(
        consumers={consumer.config.dlq_topic: consumer},  # type: ignore[dict-item]
        producer=producer,  # type: ignore[arg-type]
        quarantine_producer=quarantine,  # type: ignore[arg-type]
        tracking=None,
    )
    result = await handler._process_message(_message(""), handler._config)

    assert producer.replayed == [], (
        "the producer published a zero-byte record back onto the original topic"
    )
    assert len(quarantine.quarantined) == 1
    assert "empty" in quarantine.quarantined[0][1].lower()
    assert result.status.value == "quarantined", result.status

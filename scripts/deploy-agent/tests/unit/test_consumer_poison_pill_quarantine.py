# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An undecodable command is quarantined, never re-read forever (OMN-16442).

The failure these cover was measured, not imagined: one snappy-compressed
record on ``onex.cmd.deploy.rebuild-requested.v1`` against a ``kafka-python``
client with no snappy codec raised ``UnsupportedCodecError`` inside every
``poll()``. The agent died before committing, so each restart re-read the same
offset — twelve crashes in sixty seconds and the deploy control plane stayed
down until the record aged out.

The invariant asserted throughout: **the offset advances**. A durable record
and a dead-letter publish are how the skip stays visible, but they are
best-effort and must never be able to re-stall the partition.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from deploy_agent.consumer import (
    QUARANTINE_COMMIT_METADATA,
    DeployConsumer,
    UndecodableValue,
    deserialize_command_value,
)
from deploy_agent.events import TOPIC_DEPLOY_COMMAND_DLQ, EnumRuntimeLane
from kafka.errors import CorruptRecordError, UnsupportedCodecError
from kafka.structs import TopicPartition

_TOPIC = "onex.cmd.deploy.rebuild-requested.v1"


def _consumer(tmp_path: Path) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.kafka_config = Mock()
    consumer.kafka_config.producer_kwargs.return_value = {}
    consumer.quarantine_dir = tmp_path / "quarantine"
    return consumer


def _quarantine_records(tmp_path: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "quarantine").glob("*.json"))
    ]


# ── the deserializer never raises ────────────────────────────────────────────


@pytest.mark.unit
def test_truncated_json_yields_a_marker_not_an_exception() -> None:
    value = deserialize_command_value(b'{"correlation_id": "aaa')
    assert isinstance(value, UndecodableValue)
    assert "JSONDecodeError" in value.error


@pytest.mark.unit
def test_non_utf8_bytes_yield_a_marker() -> None:
    value = deserialize_command_value(b"\xff\xfe\x00binary")
    assert isinstance(value, UndecodableValue)
    assert value.raw_bytes == 9


@pytest.mark.unit
def test_valid_json_that_is_not_an_object_yields_a_marker() -> None:
    value = deserialize_command_value(b"[1, 2, 3]")
    assert isinstance(value, UndecodableValue)
    assert "expected a JSON object" in value.error


@pytest.mark.unit
def test_a_real_command_object_passes_straight_through() -> None:
    assert deserialize_command_value(b'{"scope": "runtime"}') == {"scope": "runtime"}


# ── record-level quarantine ──────────────────────────────────────────────────


@pytest.mark.unit
def test_undecodable_record_is_dlqd_and_the_offset_advances(tmp_path: Path) -> None:
    consumer = _consumer(tmp_path)
    msg = SimpleNamespace(
        value=deserialize_command_value(b"not json at all"),
        topic=_TOPIC,
        partition=0,
        offset=18,
        key=b"manual-abc",
    )

    producer = Mock()
    with patch("kafka.KafkaProducer", return_value=producer):
        cmd, reason = consumer._process_message(msg)

    assert cmd is None
    assert reason == "undecodable_payload"
    # THE invariant: committed, so the partition is not stalled.
    consumer.consumer.commit.assert_called_once()

    producer.send.assert_called_once()
    assert producer.send.call_args.args[0] == TOPIC_DEPLOY_COMMAND_DLQ
    dlq = producer.send.call_args.kwargs["value"]
    assert dlq["skipped_offset"] == 18
    assert dlq["committed_offset"] == 19

    (record,) = _quarantine_records(tmp_path)
    assert record["class"] == "undecodable_payload"
    assert record["topic"] == _TOPIC


@pytest.mark.unit
def test_signed_command_the_contract_refuses_is_quarantined(tmp_path: Path) -> None:
    """The exact drift shape: `reason` present, `runtime_lane` missing."""
    consumer = _consumer(tmp_path)
    msg = SimpleNamespace(
        value={
            "correlation_id": "aaaaaaaa-0000-0000-0000-000000000001",
            "git_ref": "origin/dev",
            "reason": "manual trigger by operator",
            "requested_by": "operator-manual",
            "scope": "runtime",
            "services": [],
            "_signature": "a" * 64,
        },
        topic=_TOPIC,
        partition=0,
        offset=19,
        key=None,
    )

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        patch("kafka.KafkaProducer", return_value=Mock()),
    ):
        cmd, reason = consumer._process_message(msg)

    assert cmd is None
    assert reason == "invalid_payload"
    consumer.consumer.commit.assert_called_once()

    (record,) = _quarantine_records(tmp_path)
    assert record["class"] == "invalid_payload"
    assert "runtime_lane" in record["reason"]
    assert record["committed_offset"] == 20


@pytest.mark.unit
def test_a_failing_dlq_publish_still_leaves_the_offset_advanced(
    tmp_path: Path,
) -> None:
    """Durability is best-effort; the un-stall is not."""
    consumer = _consumer(tmp_path)
    msg = SimpleNamespace(
        value=deserialize_command_value(b"{{{"),
        topic=_TOPIC,
        partition=0,
        offset=20,
        key=None,
    )

    with patch("kafka.KafkaProducer", side_effect=OSError("broker unreachable")):
        cmd, reason = consumer._process_message(msg)

    assert (cmd, reason) == (None, "undecodable_payload")
    consumer.consumer.commit.assert_called_once()
    # The local record is the surviving evidence.
    assert _quarantine_records(tmp_path)[0]["skipped_offset"] == 20


# ── fetch-level quarantine ───────────────────────────────────────────────────


def _assign(
    consumer: DeployConsumer, *, position: int, highwater: int
) -> TopicPartition:
    tp = TopicPartition(_TOPIC, 0)
    consumer.consumer.assignment.return_value = {tp}
    consumer.consumer.position.return_value = position
    consumer.consumer.highwater.return_value = highwater
    return tp


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        UnsupportedCodecError("Libraries for snappy compression codec not found"),
        CorruptRecordError("crc mismatch"),
    ],
)
def test_undecodable_fetch_seeks_past_exactly_one_offset(
    tmp_path: Path, error: Exception
) -> None:
    consumer = _consumer(tmp_path)
    tp = _assign(consumer, position=18, highwater=19)
    consumer.consumer.poll.side_effect = error

    with patch("kafka.KafkaProducer", return_value=Mock()):
        cmd, reason = consumer.poll_and_accept()

    assert cmd is None
    assert reason == "undecodable_fetch"
    consumer.consumer.seek.assert_called_once_with(tp, 19)

    committed = consumer.consumer.commit.call_args.args[0]
    assert committed[tp].offset == 19
    assert committed[tp].metadata == QUARANTINE_COMMIT_METADATA

    (record,) = _quarantine_records(tmp_path)
    assert record["class"] == "undecodable_fetch"
    assert record["advanced"] == [
        {
            "topic": _TOPIC,
            "partition": 0,
            "skipped_offset": 18,
            "committed_offset": 19,
        }
    ]


@pytest.mark.unit
def test_a_caught_up_partition_is_never_advanced(tmp_path: Path) -> None:
    """Nothing pending means nothing to skip — advancing would eat a future command."""
    consumer = _consumer(tmp_path)
    _assign(consumer, position=19, highwater=19)
    consumer.consumer.poll.side_effect = UnsupportedCodecError("snappy")

    cmd, reason = consumer.poll_and_accept()

    assert cmd is None
    assert reason == "undecodable_fetch_unrecovered"
    consumer.consumer.seek.assert_not_called()
    consumer.consumer.commit.assert_not_called()
    # Loud, not silent: the unrecovered stall is still written down.
    assert _quarantine_records(tmp_path)[0]["advanced"] == []


@pytest.mark.unit
def test_transient_errors_propagate_and_never_advance_an_offset(
    tmp_path: Path,
) -> None:
    """A broker outage must not be laundered into a skipped command."""
    consumer = _consumer(tmp_path)
    _assign(consumer, position=18, highwater=19)
    consumer.consumer.poll.side_effect = ConnectionError("broker down")

    with pytest.raises(ConnectionError):
        consumer.poll_and_accept()

    consumer.consumer.seek.assert_not_called()
    consumer.consumer.commit.assert_not_called()

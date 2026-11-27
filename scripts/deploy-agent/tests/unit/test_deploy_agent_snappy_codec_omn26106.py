# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent decodes a snappy record and never stalls on one (OMN-26106).

Measured on h201 on 2026-11-27: at 06:12:04Z the dev deploy agent's consumer
raised ``UnsupportedCodecError: Libraries for snappy compression codec not
found`` on the first fetch after a rebalance, the quarantine could not advance
(the fetch failed before a high-water mark existed), the poll loop stopped, and
the member left its group at 06:17:05Z. ``kafka-python`` decodes snappy only
through ``import snappy`` (python-snappy); the agent's own pyproject declared
neither it nor ``cramjam``.

These tests use the real codec from the agent's own environment (unit mode and
integration mode differ only in the broker, RULING 2026-11-27T03:29:55Z).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane
from kafka.codec import has_snappy
from kafka.errors import UnsupportedCodecError
from kafka.record.default_records import (
    DefaultRecordBatch,
    DefaultRecordBatchBuilder,
)
from kafka.structs import TopicPartition

_TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
_AGENT_ROOT = Path(__file__).resolve().parents[2]
_SNAPPY = 2  # kafka record attributes: compression codec id for snappy


@pytest.mark.unit
def test_agent_declares_the_snappy_codec_kafka_python_imports() -> None:
    declared = tomllib.loads((_AGENT_ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert any(d.lower().startswith("python-snappy") for d in declared), declared
    assert "python-snappy" in (_AGENT_ROOT / "uv.lock").read_text()


@pytest.mark.unit
def test_a_snappy_compressed_record_batch_decodes_in_the_agent_environment() -> None:
    assert has_snappy(), "kafka-python has no snappy codec in this environment"
    payload = json.dumps({"git_ref": "refs/heads/dev"}).encode()
    builder = DefaultRecordBatchBuilder(
        magic=2,
        compression_type=_SNAPPY,
        is_transactional=0,
        producer_id=-1,
        producer_epoch=-1,
        base_sequence=-1,
        batch_size=1024,
    )
    assert builder.append(0, timestamp=1, key=None, value=payload, headers=[])
    raw = bytes(builder.build())
    (record,) = list(DefaultRecordBatch(raw))
    assert bytes(record.value) == payload


def _consumer(tmp_path: Path) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.kafka_config = Mock()
    consumer.kafka_config.producer_kwargs.return_value = {}
    consumer.quarantine_dir = tmp_path / "quarantine"
    return consumer


@pytest.mark.unit
def test_a_fetch_that_fails_before_a_highwater_exists_is_still_skipped(
    tmp_path: Path,
) -> None:
    """Journal 06:12:04Z: position known, high-water unknown, nothing advanced."""
    consumer = _consumer(tmp_path)
    tp = TopicPartition(_TOPIC, 0)
    consumer.consumer.assignment.return_value = {tp}
    consumer.consumer.position.return_value = 18
    consumer.consumer.highwater.return_value = None
    consumer.consumer.poll.side_effect = UnsupportedCodecError("snappy")

    with patch("kafka.KafkaProducer", return_value=Mock()):
        cmd, reason = consumer.poll_and_accept()

    assert (cmd, reason) == (None, "undecodable_fetch")
    consumer.consumer.seek.assert_called_once_with(tp, 19)

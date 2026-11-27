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

import os
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane
from kafka.errors import UnsupportedCodecError
from kafka.structs import TopicPartition

_TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
_AGENT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_agent_declares_the_snappy_codec_kafka_python_imports() -> None:
    declared = tomllib.loads((_AGENT_ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert any(d.lower().startswith("python-snappy") for d in declared), declared
    assert "python-snappy" in (_AGENT_ROOT / "uv.lock").read_text()


# Run inside the agent's own locked environment. The evidence verifier and the
# repository's root suite both run this file from the ROOT project's
# environment, which does not (and must not) declare the agent's codec, so the
# decode is proven in a child process that builds the agent environment from the
# agent's own ``uv.lock`` instead of trusting whichever interpreter runs pytest.
_DECODE_IN_AGENT_ENV = """
import json
from kafka.codec import has_snappy
from kafka.record.default_records import DefaultRecordBatch, DefaultRecordBatchBuilder

assert has_snappy(), "kafka-python has no snappy codec in this environment"
payload = json.dumps({"git_ref": "refs/heads/dev"}).encode()
builder = DefaultRecordBatchBuilder(
    magic=2, compression_type=2, is_transactional=0, producer_id=-1,
    producer_epoch=-1, base_sequence=-1, batch_size=1024,
)
assert builder.append(0, timestamp=1, key=None, value=payload, headers=[])
(record,) = list(DefaultRecordBatch(bytes(builder.build())))
assert bytes(record.value) == payload
print("snappy-batch-decoded")
"""

# Variables a verifier or an activated venv sets to point uv at another
# environment; the child must resolve the agent project on its own.
_FOREIGN_ENV_VARS = (
    "UV_PROJECT_ENVIRONMENT",
    "UV_NO_SYNC",
    "UV_FROZEN",
    "VIRTUAL_ENV",
    "PYTHONPATH",
)


def _decode_in_project_env(project: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in _FOREIGN_ENV_VARS}
    return subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--no-dev",
            "--project",
            str(project),
            "python",
            "-c",
            _DECODE_IN_AGENT_ENV,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=170,
        check=False,
    )


@pytest.mark.unit
def test_a_snappy_compressed_record_batch_decodes_in_the_agent_environment() -> None:
    result = _decode_in_project_env(_AGENT_ROOT)
    assert result.returncode == 0, result.stderr[-800:]
    assert "snappy-batch-decoded" in result.stdout


@pytest.mark.unit
def test_the_decode_proof_fails_in_an_environment_without_the_codec(
    tmp_path: Path,
) -> None:
    """Known bad: the same proof over a project that omits python-snappy."""
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "pyproject.toml").write_text(
        '[project]\nname = "bare"\nversion = "0"\nrequires-python = ">=3.12"\n'
        'dependencies = ["kafka-python>=2.0.2"]\n'
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-dev",
            "--project",
            str(bare),
            "python",
            "-c",
            _DECODE_IN_AGENT_ENV,
        ],
        env={k: v for k, v in os.environ.items() if k not in _FOREIGN_ENV_VARS},
        capture_output=True,
        text=True,
        timeout=170,
        check=False,
    )
    assert result.returncode != 0
    assert "no snappy codec" in result.stderr


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

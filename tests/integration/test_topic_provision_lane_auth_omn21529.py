# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane declaration, credential store and topic contract meet at the CLI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.cli.store_lane_credential import StoreLaneCredential

pytestmark = pytest.mark.integration

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "create_kafka_topics.py"
_TOPIC = "onex.evt.landing-l18728.topic-provision-sasl.v1"


@pytest.mark.parametrize("identity", ["stored", "environment", "missing"])
def test_lane_contract_reaches_authenticated_topic_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    identity: str,
) -> None:
    """Run the CLI with real file readers; substitute only the broker boundary."""
    declaration = tmp_path / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        'lanes:\n  dev:\n    broker: "broker.invalid:19092"\n'
        "    security_protocol: SASL_PLAINTEXT\n"
        "    sasl_mechanism: SCRAM-SHA-256\n",
        encoding="utf-8",
    )
    contracts = tmp_path / "nodes" / "sample"
    contracts.mkdir(parents=True)
    (contracts / "contract.yaml").write_text(
        f"event_bus:\n  publish_topics:\n    - topic: {_TOPIC}\n"
        "      topic_config:\n        partitions: 2\n"
        "        replication_factor: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMNIBASE_PATH", str(tmp_path))
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path))
    monkeypatch.setenv("ONEX_TOPIC_PROVISIONING_PROFILE", "self_hosted")
    monkeypatch.delenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", raising=False)
    monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
    monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)
    password = "integration-only-value"
    if identity == "stored":
        StoreLaneCredential(onex_home=tmp_path / "home" / ".onex").save(
            lane="dev", sasl_username="provisioner", sasl_password=password
        )
    elif identity == "environment":
        monkeypatch.setenv("KAFKA_SASL_USERNAME", "provisioner")
        monkeypatch.setenv("KAFKA_SASL_PASSWORD", password)

    before = SimpleNamespace(brokers={1: object()}, topics={})
    after = SimpleNamespace(brokers={1: object()}, topics={_TOPIC: object()})
    admin = MagicMock()
    admin.list_topics.side_effect = [before, after, after]
    future = MagicMock()
    future.exception.return_value = None
    admin.create_topics.return_value = {_TOPIC: future}
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT), "--lane", "dev", "--contracts-root", str(contracts)],
    )
    with patch("confluent_kafka.admin.AdminClient", return_value=admin) as client:
        with pytest.raises(SystemExit) as first:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
        output = capsys.readouterr()
        assert password not in output.out + output.err
        if identity == "missing":
            assert first.value.code == 1
            assert "lanes.dev" in output.err
            assert "lane-login" in output.err
            client.assert_not_called()
            return

        assert first.value.code == 0, output.err
        client.assert_called_once_with(
            {
                "bootstrap.servers": "broker.invalid:19092",
                "security.protocol": "SASL_PLAINTEXT",
                "sasl.mechanism": "SCRAM-SHA-256",
                "sasl.username": "provisioner",
                "sasl.password": password,
            }
        )
        (topics,) = admin.create_topics.call_args.args
        assert len(topics) == 1
        assert topics[0].topic == _TOPIC
        assert topics[0].num_partitions == 2
        assert topics[0].replication_factor == 1
        assert "Result: 1 topics created successfully." in output.out

        with pytest.raises(SystemExit) as second:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
        assert second.value.code == 0
        assert "All 1 topics already exist" in capsys.readouterr().out
        admin.create_topics.assert_called_once()

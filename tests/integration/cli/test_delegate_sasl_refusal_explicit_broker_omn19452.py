# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""An explicit broker address that refuses SASL names the lane login (OMN-19452 AC1).

The ticket's measured case: the operator passed a broker address directly, with
no lane selected, to a broker that requires SASL. The run ended in "Fix the
broker address", which was wrong, because the address was right and the login
was not. ``test_delegate_sasl_refusal_cli_omn19452.py`` covers a selected lane.
This module covers the explicit address with no lane. The refusal must be typed
``sasl_refused`` and must name both ``onex auth lane-login`` and ``--lane``.

All names below are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from tests.helpers.fake_sasl_refusing_broker import (
    SASL_MECHANISM,
    serve_sasl_refusing_broker,
)

pytestmark = pytest.mark.integration

_PRINCIPAL = "explicit-broker-synthetic-host"
_BROKER_ADVICE = "Fix the broker address"
_ORCHESTRATOR_CONTRACT = (
    "---\n"
    "name: node_delegate_skill_orchestrator\n"
    "event_bus:\n"
    "  subscribe_topics:\n"
    "    - onex.cmd.omnimarket.delegate-skill.v1\n"
)


@pytest.fixture(autouse=True)
def _isolated_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No workspace or lane; a SASL identity only in the environment."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", SASL_MECHANISM)
    monkeypatch.setenv("KAFKA_SASL_USERNAME", _PRINCIPAL)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "synthetic-not-real")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_ORCHESTRATOR_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))


def test_an_explicit_address_refusing_sasl_names_the_lane_login(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    with serve_sasl_refusing_broker() as broker:
        result = CliRunner().invoke(
            delegate_command,
            [
                "Reply with exactly the word READY",
                "--task-type",
                "document",
                "--bus",
                "kafka",
                "--kafka-bootstrap",
                broker,
                "--locus",
                "deployed-lane",
                "--state-root",
                str(state_root),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code != 0
    output = str(result.output)
    assert "onex auth lane-login" in output
    assert "--lane" in output
    assert _BROKER_ADVICE not in output

    written = sorted(state_root.glob("runs/*/receipt.json"))
    assert len(written) == 1, written
    receipt = json.loads(written[0].read_text(encoding="utf-8"))
    refusal = receipt["transport_refusal"]
    assert refusal["reason"] == "sasl_refused"
    assert refusal["transport_error_type"] == "DelegateLocusSaslRefusedError"
    assert "onex auth lane-login" in refusal["remediation"]
    assert "--lane" in refusal["remediation"]
    assert "synthetic-not-real" not in json.dumps(receipt)

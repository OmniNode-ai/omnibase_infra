# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` names the lane login when a SASL lane refuses it (OMN-19452 AC1).

End-to-end through the real command: the real option parsing, the real lane
declaration read from disk, the real by-reference credential store under a
throwaway home, the real aiokafka admin client, and a listener that refuses the
SASL login on the wire. Before this change the run ended in::

    ... Fix the broker address, or pass --bus inmemory --locus in-process ...

which points at a broker address that was never wrong. The refusal now says
whose login the broker refused, and names the two things that fix it:
``onex auth lane-login`` and ``--lane``.

All names below are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from tests.helpers.fake_sasl_refusing_broker import (
    SASL_MECHANISM,
    serve_sasl_refusing_broker,
)

pytestmark = pytest.mark.integration

_LANE = "dev"
_PRINCIPAL = "dev-cli-synthetic-host"
_BROKER_ADVICE = "Fix the broker address"

#: Stands in for the omnimarket-provided orchestrator contract, which this repo
#: cannot resolve by layering. Only its name and command topic are read: the
#: locus probe refuses before anything is dispatched, so no handler runs.
_ORCHESTRATOR_CONTRACT = (
    "---\n"
    "name: node_delegate_skill_orchestrator\n"
    "event_bus:\n"
    "  subscribe_topics:\n"
    "    - onex.cmd.omnimarket.delegate-skill.v1\n"
)


@pytest.fixture(autouse=True)
def _isolated_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient workspace, drift guard, SASL environment or home directory."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    for name in (
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_ORCHESTRATOR_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))


def _workspace(root: Path, *, broker: str) -> Path:
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(
        "lanes:\n"
        f"  {_LANE}:\n"
        f'    broker: "{broker}"\n'
        "    security_protocol: SASL_PLAINTEXT\n"
        f"    sasl_mechanism: {SASL_MECHANISM}\n",
        encoding="utf-8",
    )
    return root


def _store_identity(tmp_path: Path) -> None:
    StoreLaneCredential(onex_home=tmp_path / "home" / ".onex").save(
        lane=_LANE, sasl_username=_PRINCIPAL, sasl_password="synthetic-not-real"
    )


def _invoke(*, tmp_path: Path, broker: str) -> tuple[object, Path]:
    state_root = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly the word READY",
            "--task-type",
            "document",
            "--bus",
            "kafka",
            "--lane",
            _LANE,
            "--locus",
            "deployed-lane",
            "--omnibase-path",
            str(_workspace(tmp_path / "workspace", broker=broker)),
            "--state-root",
            str(state_root),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )
    return result, state_root


def _sole_receipt(state_root: Path) -> dict[str, object]:
    written = sorted(state_root.glob("runs/*/receipt.json"))
    assert len(written) == 1, f"expected exactly one receipt.json, got {written}"
    parsed = json.loads(written[0].read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


class TestASaslRefusalNamesTheLaneLogin:
    def test_the_operator_is_told_the_login_not_the_address(
        self, tmp_path: Path
    ) -> None:
        _store_identity(tmp_path)
        with serve_sasl_refusing_broker() as broker:
            result, _state_root = _invoke(tmp_path=tmp_path, broker=broker)

        assert result.exit_code != 0
        text = str(result.output)
        assert "onex auth lane-login" in text
        assert f"--lane {_LANE}" in text
        assert _PRINCIPAL in text
        assert _BROKER_ADVICE not in text

    def test_the_receipt_carries_a_typed_sasl_refusal(self, tmp_path: Path) -> None:
        _store_identity(tmp_path)
        with serve_sasl_refusing_broker() as broker:
            _result, state_root = _invoke(tmp_path=tmp_path, broker=broker)

        receipt = _sole_receipt(state_root)
        assert receipt["status"] == "failed"
        assert receipt["terminal_class"] == "transport"
        refusal = receipt["transport_refusal"]
        assert isinstance(refusal, dict)
        assert refusal["reason"] == "sasl_refused"
        assert refusal["transport_error_type"] == "DelegateLocusSaslRefusedError"
        assert _PRINCIPAL in str(refusal["transport_error"])
        assert "REDACTED" not in str(refusal["transport_error"])
        remediation = str(refusal["remediation"])
        assert "onex auth lane-login" in remediation
        assert f"--lane {_LANE}" in remediation
        assert "synthetic-not-real" not in json.dumps(receipt)

    def test_no_route_identity_is_synthesised(self, tmp_path: Path) -> None:
        """A refused login means no rung ran: same fail-closed attribution."""
        _store_identity(tmp_path)
        with serve_sasl_refusing_broker() as broker:
            _result, state_root = _invoke(tmp_path=tmp_path, broker=broker)

        receipt = _sole_receipt(state_root)
        assert receipt["route_attributed"] is False
        assert receipt["attempts"] == []

    def test_the_refusal_receipt_times_the_phases_it_reached(
        self, tmp_path: Path
    ) -> None:
        """Startup and the probe ran; nothing was connected, subscribed or published."""
        _store_identity(tmp_path)
        with serve_sasl_refusing_broker() as broker:
            _result, state_root = _invoke(tmp_path=tmp_path, broker=broker)

        durations = _sole_receipt(state_root)["phase_durations"]
        assert isinstance(durations, dict)
        assert isinstance(durations["startup_seconds"], float)
        assert isinstance(durations["locus_probe_seconds"], float)
        for field in (
            "bus_connect_seconds",
            "reply_subscribe_seconds",
            "publish_seconds",
            "terminal_wait_seconds",
        ):
            assert durations[field] is None, field

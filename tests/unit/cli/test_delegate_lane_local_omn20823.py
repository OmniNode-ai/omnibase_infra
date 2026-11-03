# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20823 AC6: ``onex delegate --bus kafka --lane local`` with no stored login.

After ``onex auth lane-logout --lane local`` a developer machine holds no
identity for the ``local`` lane, which declares SASL_PLAINTEXT. With
``KAFKA_SASL_USERNAME`` and ``KAFKA_SASL_PASSWORD`` unset and ``ONEX_TENANT_ID``
set, the command is refused before anything is dispatched, naming
``onex auth lane-login``. With the two ambient variables both set, the
supported fallback of ``delegate_lane_credentials`` answers instead: the run is
dispatched and no stored identity is bound.

The command runs through its click surface. ``run_receipt_mode`` is replaced by
a recorder, so "dispatched" means the runtime was handed the run, and an empty
recorder means nothing connected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bound_lane_client_transport,
)
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)

pytestmark = pytest.mark.unit

LOCAL_BROKER = "localhost:19092"

# The `local` lane as the workspace lane table declares it (OMN-20824).
_DECLARATION = f"""
lanes:
  local:
    broker: "{LOCAL_BROKER}"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
"""


class _Machine:
    """A developer machine: a home, a workspace declaring lane local, a recorder."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.captured: dict[str, object] = {}
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.workspace = tmp_path / "ws"
        declaration = (
            self.workspace
            / ".onex"
            / "workspace-config"
            / LANE_DECLARATION_RELATIVE_PATH
        )
        declaration.parent.mkdir(parents=True)
        declaration.write_text(_DECLARATION, encoding="utf-8")

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            self.captured.update(kwargs)
            self.captured["lane_transport"] = bound_lane_client_transport()
            return 0

        monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: self.home))
        monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
        monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)
        monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
        monkeypatch.setenv("ONEX_TENANT_ID", "local-0a1b2c3d4e5f")

    def store(self) -> StoreLaneCredential:
        return StoreLaneCredential(onex_home=self.home / ".onex")

    def delegate(self) -> Result:
        return CliRunner().invoke(
            delegate_command,
            [
                "Reply with exactly one word: hello",
                "--task-type",
                "document",
                "--bus",
                "kafka",
                "--lane",
                "local",
                "--locus",
                "in-process",
                "--omnibase-path",
                str(self.workspace),
                "--state-root",
                str(self.tmp / "state"),
            ],
        )


@pytest.fixture
def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Machine:
    return _Machine(tmp_path, monkeypatch)


def test_after_logout_the_local_lane_is_refused_before_it_connects(
    machine: _Machine,
) -> None:
    machine.store().save(
        lane="local", sasl_username="dev-person-mac", sasl_password="stored-value"
    )
    machine.store().clear("local")
    result = machine.delegate()
    assert result.exit_code != 0, result.output
    output = " ".join(result.output.split())
    assert "onex auth lane-login --lane local" in output
    assert "SASL_PLAINTEXT" in output
    assert machine.captured == {}, "the run was dispatched without an identity"


def test_with_the_ambient_pair_set_the_fallback_answers(
    machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "ambient-user")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "ambient-value")
    result = machine.delegate()
    assert result.exit_code == 0, result.output
    assert machine.captured["backend_overrides"] == {
        "event_bus": "kafka",
        "kafka_bootstrap": LOCAL_BROKER,
    }
    # No stored identity is bound: the environment-sourced client construction,
    # which reads the ambient pair, answers.
    assert machine.captured["lane_transport"] is None


def test_a_stored_login_is_bound_for_the_local_lane(machine: _Machine) -> None:
    """Positive control: the same machine with a login is dispatched and bound."""
    machine.store().save(
        lane="local", sasl_username="dev-person-mac", sasl_password="stored-value"
    )
    result = machine.delegate()
    assert result.exit_code == 0, result.output
    transport = machine.captured["lane_transport"]
    assert isinstance(transport, ModelLaneClientTransport)
    assert transport.sasl_username == "dev-person-mac"
    assert transport.bootstrap_servers == LOCAL_BROKER

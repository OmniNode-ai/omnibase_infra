# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev deploy agent's STATUS append runs without the agent's own broker declarations (OMN-26316).

On omni-201 the dev deploy agent logged ``Dev boot STATUS delivery still pending: append
command refused`` about every three seconds from 2026-11-23T21:55 EST. The bus append
command (``python -m omnimarket.work_ledger_bus.cli append --bus-lane dev``) exited 1
before it reached the broker: the agent's unit declares ``KAFKA_SASL_MECHANISM`` and
``KAFKA_SECURITY_PROTOCOL`` for the agent's own consumer, with the lane login held only
under the ``DEV_`` prefix, and ``ModelKafkaEventBusConfig.default()`` rejects a SASL
mechanism with no ``KAFKA_SASL_USERNAME`` / ``KAFKA_SASL_PASSWORD`` before the lane's stored
identity is bound. The same command with those declarations absent was accepted by the
broker and the ledger (receipt ``status=accepted`` from the ledger host).

The stub below fails the way ``default()`` fails: it exits 1 when the agent's SASL
declaration reaches it, and prints the accepted receipt when it does not.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard

pytestmark = pytest.mark.unit

_AGENT_BROKER_DECLARATION = {
    "KAFKA_SASL_MECHANISM": "SCRAM-SHA-256",
    "KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
    "KAFKA_SASL_ENV_PREFIX": "DEV_",
}
_UNRELATED_NAME = "ONEX_LEDGER_HOST_NAME"

_STUB = f"""#!{sys.executable}
import json
import os
import sys

if os.environ.get("KAFKA_SASL_MECHANISM"):
    sys.stderr.write(
        "ProtocolConfigurationError: sasl_mechanism='SCRAM-SHA-256' requires non-empty "
        "credential fields: sasl_plain_username, sasl_plain_password\\n"
    )
    sys.exit(1)
if os.environ.get({_UNRELATED_NAME!r}) != "h201":
    sys.stderr.write("the host name the unit declares did not reach the command\\n")
    sys.exit(1)
sys.stdin.read()
print(json.dumps({{"status": "accepted", "exit_code": 0}}))
"""


class _FakeExecution(types.ModuleType):
    """The slice of ``deploy_agent.executor`` the append uses: ``_run`` and ``subprocess``."""

    subprocess = subprocess

    @staticmethod
    def _run(
        cmd: list[str],
        timeout: int,
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            env=env,
        )


def _execution() -> types.ModuleType:
    return _FakeExecution("fake_execution")


@pytest.fixture
def agent_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = tmp_path / "append_stub.py"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("ONEX_LEDGER_BUS_APPEND_COMMAND", str(stub))
    monkeypatch.setenv(_UNRELATED_NAME, "h201")
    for name, value in _AGENT_BROKER_DECLARATION.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DEV_KAFKA_SASL_USERNAME", "dev-lane-login")
    monkeypatch.setenv("DEV_KAFKA_SASL_PASSWORD", "not-a-real-value")


def test_status_append_is_accepted_under_the_agents_sasl_declaration(
    agent_environment: None,
) -> None:
    delivered = guard._append_status_over_bus(
        _execution(),
        "2026-11-24T06:00:00Z | STATUS | lane=deploy-agent-dev | row",
        lane="deploy-agent-dev",
        request_id="6a0a3f2e-2d0f-5a43-9d4c-0e6b0c1d2f11",
    )

    assert delivered is True


def test_stub_refuses_when_the_declaration_reaches_it(
    agent_environment: None,
) -> None:
    """Positive control: the stub fails as the real command did, so the pass above is earned."""
    stub = os.environ["ONEX_LEDGER_BUS_APPEND_COMMAND"]

    refused = subprocess.run(
        [stub, "id"], input="row\n", capture_output=True, text=True, check=False
    )

    assert refused.returncode == 1
    assert "requires non-empty credential fields" in refused.stderr


def test_refusal_log_names_the_commands_last_stderr_line(
    agent_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The journal line carried no answer, so three thousand refusals read as one unexplained word."""
    monkeypatch.setenv("ONEX_LEDGER_HOST_NAME", "not-h201")

    delivered = guard._append_status_over_bus(
        _execution(),
        "2026-11-24T06:00:00Z | STATUS | lane=deploy-agent-dev | row",
        lane="deploy-agent-dev",
        request_id="6a0a3f2e-2d0f-5a43-9d4c-0e6b0c1d2f11",
    )

    assert delivered is False
    assert "append command refused" in caplog.text
    assert "the host name the unit declares did not reach the command" in caplog.text

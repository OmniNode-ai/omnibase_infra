# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` classes a wait that ends while the command is queued (OMN-22211).

Drives the real click entry point, the real hard-timeout backstop, the real
receipt writer and the real in-memory bus against a contract whose handler
blocks. Only the broker queue-depth probe is replaced, because no broker is
present: it reports the backlog the deployed run 91fda261 observed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.model_delegate_queue_depth import ModelDelegateQueueDepth
from tests.helpers.cli_registry_stand_in import (
    use_stand_in_registry,
    wiring_authority,
)

_BLOCKING_NOOP_CONTRACT = (
    "---\n"
    "name: blocking_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.blocking-noop-completed.v1\n"
    "handler:\n"
    "  module: tests.fixtures.handler_blocking_noop\n"
    "  class: HandlerBlockingNoop\n"
    "  input_model: tests.fixtures.handler_blocking_noop"
    ".ModelBlockingNoopRequest\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.handler_blocking_noop"
    ":HandlerBlockingNoop\n"
)


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records_ahead: int | None
) -> dict[str, object]:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_BLOCKING_NOOP_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    use_stand_in_registry(
        monkeypatch, wiring_authority(terminal_delivery_margin_seconds=1)
    )
    monkeypatch.setattr(cli_delegate, "_HARD_TIMEOUT_GRACE_SECONDS", 1)
    monkeypatch.setattr(
        cli_delegate,
        "observe_delegate_queue_depth",
        lambda **_kwargs: ModelDelegateQueueDepth(
            records_ahead=records_ahead,
            backlog_records=None if records_ahead is None else records_ahead + 1,
            consumer_group="delegate-consumer-group",
        ),
    )
    state_root = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            "List the first five prime numbers",
            "--json",
            "--state-root",
            str(state_root),
            "--timeout",
            "1",
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    runs = list((state_root / "runs").iterdir())
    assert len(runs) == 1
    receipt: dict[str, object] = json.loads(
        (runs[0] / "receipt.json").read_text(encoding="utf-8")
    )
    return receipt


@pytest.mark.integration
def test_backlog_ahead_of_command_records_queued_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _run(tmp_path, monkeypatch, 50)

    assert receipt["terminal_class"] == "queued_not_started"
    assert receipt["terminal_recorded"] is False
    assert receipt["attempts"] == []


@pytest.mark.integration
def test_empty_backlog_stays_a_bare_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _run(tmp_path, monkeypatch, 0)

    assert receipt["terminal_class"] == "timeout"

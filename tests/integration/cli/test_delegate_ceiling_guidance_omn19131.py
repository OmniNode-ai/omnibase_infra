# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A runtime ceiling refusal reaches the operator with actionable flag guidance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_core.models.delegation.wire import ModelDelegationBudgetRefusal
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.delegate_caller import CALLER_LANE_ENV_VARS
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
    ModelDelegateSkillFixtureTerminal,
)

pytestmark = pytest.mark.integration


class HandlerCeilingNoop(HandlerCorrelatedNoop):
    """Return a typed refusal above the fixture's 240-second execution ceiling."""

    def handle(
        self, request: ModelCorrelatedNoopRequest
    ) -> ModelDelegateSkillFixtureTerminal:
        requested = request.requested_timeout_seconds
        if requested is not None and requested > 240:
            return ModelDelegateSkillFixtureTerminal(
                attempts=(),
                status="failed",
                budget_refusal=ModelDelegationBudgetRefusal(
                    reason="timeout_exceeds_task_class_ceiling",
                    task_type=request.task_type,
                    requested_timeout_seconds=requested,
                    task_class_timeout_ceiling_seconds=240,
                ),
                error_message=f"requested_timeout_seconds={requested} exceeds ceiling",
            )
        return super().handle(request)


@pytest.fixture
def ceiling_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real runtime and bus with a deterministic ceiling-checking handler."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract.parent.mkdir()
    contract.write_text(
        "name: ceiling_noop\n"
        "node_type: compute\n"
        "terminal_event: onex.evt.proof.ceiling-noop-completed.v1\n"
        "input_model: tests.fixtures.handler_correlated_noop.ModelCorrelatedNoopRequest\n"
        "handler:\n"
        f"  module: {__name__}\n"
        "  class: HandlerCeilingNoop\n"
        "  input_model: tests.fixtures.handler_correlated_noop.ModelCorrelatedNoopRequest\n"
        "handler_routing:\n"
        f"  default_handler: {__name__}:HandlerCeilingNoop\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _: contract)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    for name in (*CALLER_LANE_ENV_VARS, "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("ceiling_contract")
@pytest.mark.parametrize("output_mode", ["--human", "--json"])
def test_runtime_refusal_names_timeout_flag_and_ceiling(
    tmp_path: Path, output_mode: str
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        common = [
            "Reply READY",
            "--task-type",
            "document",
            "--bus",
            "inmemory",
            "--locus",
            "in-process",
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
            output_mode,
        ]
        control = runner.invoke(
            cli_delegate.delegate_command,
            [*common, "--timeout", "240", "--state-root", str(tmp_path / "control")],
            catch_exceptions=False,
        )
        assert control.exit_code == 0, control.stderr
        state = tmp_path / "refused"
        result = runner.invoke(
            cli_delegate.delegate_command,
            [*common, "--timeout", "300", "--state-root", str(state)],
            catch_exceptions=False,
        )

    assert result.exit_code == 1, result.stderr
    receipts = list((state / "runs").glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    failures = [
        line for line in result.stderr.splitlines() if "onex delegate failed:" in line
    ]
    assert len(failures) == 1, result.stderr
    line = failures[0]
    assert "`--timeout` 300 (`requested_timeout_seconds`)" in line
    assert "240s ceiling for task type `document`" in line
    assert "`--timeout` of 240 or less, or omit it" in line
    assert receipt["run_id"] in line
    assert str(receipts[0]) in line
    if output_mode == "--json":
        assert json.loads(result.stdout)["run_id"] == receipt["run_id"]
    else:
        assert result.stdout == ""

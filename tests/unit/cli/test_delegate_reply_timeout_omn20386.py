# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A deployed-lane reply that never arrives is a typed timeout, not a usage error (OMN-20386).

Measured: ``onex delegate --bus kafka --lane dev --locus deployed-lane`` whose
terminal never arrived exited 2, printed the ``Usage:`` banner and left a run
folder holding only ``workflow_result.json``. The receipt validator caught the
pre-publish refusal and nothing else, so the resolver's
``DelegateTerminalUnresolvedError`` (a ``ValueError``) left the receipt layer,
and the command maps every ``ValueError`` to ``click.UsageError``: exit 2, the
banner, and a writer that never ran.

The command and ``run_receipt_mode`` here are the real ones. Only the runtime is
replaced, by one that records what the live runtime records when the lane
accepted the command and never replied: ``result: timeout``, the wire
correlation id of the command it published, and no terminal payload.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
from omnibase_infra.cli import cli_delegate, receipt_mode
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.unit

_CONTRACT = "name: silent_lane\nnode_type: compute\n"


class _SilentLaneRuntime:
    """Published the command to the lane; the terminal never came back."""

    def __init__(self, **kwargs: object) -> None:
        state_root = kwargs["state_root"]
        input_path = kwargs["input_path"]
        run_id = kwargs["run_id"]
        assert isinstance(state_root, Path)
        assert isinstance(input_path, Path)
        self._state_root = state_root
        self._run_id = run_id
        self._wire_correlation_id = json.loads(input_path.read_text(encoding="utf-8"))[
            "correlation_id"
        ]

    def run(self) -> EnumWorkflowResult:
        self._state_root.mkdir(parents=True, exist_ok=True)
        (self._state_root / "workflow_result.json").write_text(
            json.dumps(
                {
                    "result": "timeout",
                    "exit_code": 1,
                    "workflow": "node_delegate_skill_orchestrator/contract.yaml",
                    "run_id": str(self._run_id),
                    "handler_locus": "dispatched",
                    "wire_correlation_id": self._wire_correlation_id,
                }
            ),
            encoding="utf-8",
        )
        return EnumWorkflowResult.TIMEOUT

    @property
    def exit_code(self) -> int:
        return 1

    @property
    def handler_result(self) -> object | None:
        return None


@pytest.fixture
def silent_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract.parent.mkdir()
    contract.write_text(_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: contract)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        receipt_mode, "_runtime_factory", lambda _sw: _SilentLaneRuntime
    )
    return tmp_path / "state"


def _invoke(tmp_path: Path, state_root: Path) -> tuple[Result, str]:
    result = CliRunner().invoke(
        delegate_command,
        [
            "List the first five prime numbers",
            "--state-root",
            str(state_root),
            "--timeout",
            "1",
            "--json",
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )
    run_ids = [p.name for p in (state_root / "runs").iterdir()]
    assert len(run_ids) == 1
    return result, run_ids[0]


class TestSilentLaneIsATypedTimeout:
    def test_exit_is_not_the_usage_exit_and_no_banner(
        self, tmp_path: Path, silent_lane: Path
    ) -> None:
        result, _ = _invoke(tmp_path, silent_lane)
        assert result.exit_code not in (0, 2)
        assert "Usage:" not in result.stderr

    def test_stderr_names_the_cause_and_the_run_id(
        self, tmp_path: Path, silent_lane: Path
    ) -> None:
        result, run_id = _invoke(tmp_path, silent_lane)
        stderr = result.stderr
        assert "timeout" in stderr
        assert run_id in stderr

    def test_run_folder_holds_a_receipt_recording_the_timeout(
        self, tmp_path: Path, silent_lane: Path
    ) -> None:
        result, run_id = _invoke(tmp_path, silent_lane)
        receipt = json.loads(
            (silent_lane / "runs" / run_id / "receipt.json").read_text("utf-8")
        )
        assert receipt["terminal_class"] == "timeout"
        assert receipt["run_id"] == run_id
        wire = json.loads(result.stdout.strip())["correlation_id"]
        assert receipt["correlation_id"] == wire
        assert receipt["wire_correlation_id"] == wire
        uuid.UUID(receipt["wire_correlation_id"])

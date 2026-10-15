# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result-only output across the runtime, artifact writer and OS pipes (OMN-21088)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_caller import CALLER_LANE_ENV_VARS

pytestmark = pytest.mark.integration

_CONTRACT = """---
name: correlated_noop
node_type: compute
terminal_event: onex.evt.proof.correlated-noop-completed.v1
handler:
  module: tests.fixtures.handler_correlated_noop
  class: HandlerCorrelatedNoop
  input_model: tests.fixtures.handler_correlated_noop.ModelCorrelatedNoopRequest
handler_routing:
  default_handler: tests.fixtures.handler_correlated_noop:HandlerCorrelatedNoop
"""


@pytest.mark.parametrize("answer", ["4", "line one\n\n  indented é中\nlast line"])
def test_real_runtime_emits_only_the_answer_and_files_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    """Only registry discovery is substituted; dispatch and writers are real."""
    for name in (
        "OMNI_HOME",
        "ONEX_STATE_DIR",
        "ONEX_CONTRACTS_DIR",
        "KAFKA_BOOTSTRAP_SERVERS",
        "CLAUDE_CODE_SESSION_ID",
        "ONEX_LANE_REGISTRY_ROOT",
        *CALLER_LANE_ENV_VARS,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    state = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            answer,
            "--result-only",
            "--task-type",
            "summarization",
            "--bus",
            "inmemory",
            "--locus",
            "in-process",
            "--state-root",
            str(state),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout_bytes == answer.encode("utf-8")
    assert result.stderr_bytes == b""
    (run_dir,) = (state / "runs").iterdir()
    assert (run_dir / "result.txt").read_bytes() == answer.encode("utf-8")
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "success"
    assert receipt["model"] == "fixture"
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["locus"] == "in-process"
    assert metadata["correlation_id"] == receipt["correlation_id"]
    diagnostics = (run_dir / "diagnostics.log").read_text(encoding="utf-8")
    assert "transport:" in diagnostics
    assert "delegate artifacts:" in diagnostics

    # Exercise the original forms after capture restored the process streams.
    for mode in (None, "--human", "--json"):
        original_state = tmp_path / (mode or "default")
        original = CliRunner().invoke(
            delegate_command,
            [
                answer,
                *([mode] if mode else []),
                "--task-type",
                "summarization",
                "--bus",
                "inmemory",
                "--locus",
                "in-process",
                "--state-root",
                str(original_state),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )
        assert original.exit_code == 0, original.stderr
        assert "task class:" in original.stderr
        if mode == "--json":
            envelope = json.loads(original.stdout)
            assert envelope["status"] == "success"
            assert envelope["correlation_id"]
        else:
            assert original.stdout.startswith(answer + "\n")
        (original_run,) = (original_state / "runs").iterdir()
        assert (original_run / "result.txt").read_bytes() == answer.encode("utf-8")
        assert not (original_run / "diagnostics.log").exists()


def test_installed_cli_refusal_uses_one_stderr_line_on_real_pipes(
    tmp_path: Path,
) -> None:
    """No runtime or command patch: exercise the installed entrypoint and Click."""
    done = subprocess.run(
        [
            str(Path(sys.executable).with_name("onex")),
            "delegate",
            "probe",
            "--result-only",
            "--state-root",
            str(tmp_path),
            "--no-such-option-omn21088",
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert done.returncode == 2, done.stderr
    assert done.stdout == b""
    (run_dir,) = (tmp_path / "runs").iterdir()
    receipt_path = (run_dir / "receipt.json").resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["refusal"]["stage"] == "argument_parsing"
    assert done.stderr.decode("utf-8") == (
        f"Error: argument_parsing: {receipt['refusal']['message']} "
        f"(receipt: {receipt_path})\n"
    )
    assert (run_dir / "result.txt").read_bytes() == b""
    assert (run_dir / "diagnostics.log").is_file()
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == run_dir.name

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --result-only`` hands the caller the answer and nothing else (OMN-21088).

A customer who runs ``/onex:delegate`` in Claude Code has every byte of stdout
*and* stderr land in the agent's context. ``--result-only`` makes the contract:

* success, exit 0: stdout is ``result.response`` exactly and stderr is empty;
* failure, exit non-zero: stdout is empty and stderr is exactly one line,
  ``Error: <cause>: <message> (receipt: <abs path to receipt.json>)``, for every
  way the command can end, refusals before dispatch included;
* every suppressed line goes to ``runs/<run_id>/diagnostics.log``, and
  ``receipt.json``, ``result.txt`` and ``run.json`` are still written.

Every test drives the real ``delegate_command`` entrypoint with
``run_receipt_mode`` replaced by a stand-in that behaves like the real one: it
prints its diagnostics, renders the receipt and hands the CLI's own receipt
callback the receipt, so the files are written by the production writers.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

import click
import pytest
from click.testing import CliRunner, Result

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    DelegateTenantRefusedError,
    delegate_command,
)
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.unit

_RESULT_MODEL = (
    "omnimarket.models.delegation.wire."
    "model_delegate_skill_response.ModelDelegateSkillCompleted"
)
_FAILED_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18306"
    / "failed_no_accepted_attempt_receipt.json"
)
_ERROR_LINE = re.compile(
    r"^Error: (?P<cause>[a-z_]+): (?P<message>[^\n]+) "
    r"\(receipt: (?P<path>/[^\n]+/runs/[0-9a-f-]{36}/receipt\.json)\)\n$"
)
_DIAGNOSTIC_NOISE = "stand-in runtime diagnostics"


def _receipt(response: str, run_id: uuid.UUID) -> ModelSkillResult[dict[str, object]]:
    correlation_id = uuid.uuid4()
    return ModelSkillResult[dict[str, object]](
        skill_name="delegate",
        node_name="node_delegate_skill_orchestrator",
        status=EnumSkillResultStatus.SUCCESS,
        correlation_id=correlation_id,
        run_id=run_id,
        exit_code=0,
        duration_ms=12,
        result={
            "correlation_id": str(correlation_id),
            "task_type": "research",
            "model_name": "qwen3.8",
            "provider": "http://local.invalid/v1/chat/completions",
            "response": response,
            "attempts": [
                {
                    "tier": "local",
                    "backend_id": "local-coder",
                    "model_id": "qwen3.8",
                    "quality_gate_passed": True,
                    "acceptance_decision": "accept",
                }
            ],
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


class _EnvelopeReceipt:
    """A recorded receipt, handed to the CLI the way the runtime hands its own."""

    def __init__(self, envelope: dict[str, object]) -> None:
        self.envelope = envelope

    def model_dump(self, *, mode: str) -> dict[str, object]:
        return self.envelope

    def model_dump_json(self) -> str:
        return json.dumps(self.envelope, separators=(",", ":"))


def _noisy_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    response: str = "4",
    failed: bool = False,
) -> None:
    """Replace the runtime with a stand-in that is as chatty as the real one."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("name: x\n", encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _n: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

    def _fake_run_receipt_mode(
        *,
        receipt_renderer: Callable[[object], bool] | None,
        receipt_callback: Callable[..., object],
        run_id: uuid.UUID,
        **_kwargs: object,
    ) -> int:
        click.echo(f"{_DIAGNOSTIC_NOISE}: identity line", err=True)
        click.echo(f"{_DIAGNOSTIC_NOISE}: stdout chatter")
        sys.stderr.write(f"{_DIAGNOSTIC_NOISE}: raw sys.stderr write\n")
        os.write(2, f"{_DIAGNOSTIC_NOISE}: native fd 2 write\n".encode())
        if failed:
            envelope = json.loads(_FAILED_FIXTURE.read_text(encoding="utf-8"))
            envelope["run_id"] = str(run_id)
            receipt: object = _EnvelopeReceipt(envelope)
            exit_code = int(envelope["exit_code"])
        else:
            receipt = _receipt(response, run_id)
            exit_code = 0
        assert receipt_renderer is not None
        if not receipt_renderer(receipt) and exit_code == 0:
            exit_code = 1
        receipt_callback(receipt)
        return exit_code

    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)


def _invoke(tmp_path: Path, *args: str, prompt: str | None = "what is 2+2") -> Result:
    state_root = tmp_path / "state"
    argv = [prompt] if prompt is not None else []
    return CliRunner().invoke(
        delegate_command,
        [
            *argv,
            "--result-only",
            "--bus",
            "inmemory",
            "--locus",
            "in-process",
            "--state-root",
            str(state_root),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
            *args,
        ],
        catch_exceptions=False,
    )


def _only_run_dir(tmp_path: Path) -> Path:
    runs = [p for p in (tmp_path / "state" / "runs").iterdir() if p.is_dir()]
    assert len(runs) == 1, runs
    return runs[0]


def _assert_single_error_line(result: Result, tmp_path: Path) -> re.Match[str]:
    assert result.exit_code != 0, result.stderr
    assert result.stdout == ""
    match = _ERROR_LINE.match(result.stderr)
    assert match is not None, f"not exactly one Error line: {result.stderr!r}"
    receipt = Path(match["path"])
    assert receipt.is_file(), "the line must name a receipt that exists"
    assert receipt == _only_run_dir(tmp_path).resolve() / "receipt.json"
    return match


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("ONEX_STATE_DIR", raising=False)


class TestSuccess:
    def test_stdout_is_the_response_byte_exact_and_stderr_is_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path, response="4")
        result = _invoke(tmp_path)
        assert result.exit_code == 0, result.stderr
        assert result.stdout_bytes == b"4"
        assert result.stderr == ""

    def test_a_multiline_unicode_response_is_not_altered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = "line one\n\n  indented é中\nlast line"
        _noisy_runtime(monkeypatch, tmp_path, response=response)
        result = _invoke(tmp_path)
        assert result.exit_code == 0, result.stderr
        assert result.stdout_bytes == response.encode("utf-8")
        assert result.stderr == ""

    def test_suppressed_diagnostics_land_in_the_run_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)
        result = _invoke(tmp_path)
        assert result.exit_code == 0, result.stderr
        log = (_only_run_dir(tmp_path) / "diagnostics.log").read_text(encoding="utf-8")
        for expected in (
            "task class:",
            "transport:",
            f"{_DIAGNOSTIC_NOISE}: identity line",
            f"{_DIAGNOSTIC_NOISE}: stdout chatter",
            f"{_DIAGNOSTIC_NOISE}: raw sys.stderr write",
            f"{_DIAGNOSTIC_NOISE}: native fd 2 write",
            "delegate artifacts:",
        ):
            assert expected in log, expected

    def test_the_run_files_are_still_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path, response="4")
        _invoke(tmp_path)
        run_dir = _only_run_dir(tmp_path)
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == "4"
        assert json.loads((run_dir / "receipt.json").read_text())["status"] == "success"
        assert (run_dir / "run.json").is_file()

    def test_the_default_form_is_unchanged_without_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path, response="4")
        result = CliRunner().invoke(
            delegate_command,
            [
                "what is 2+2",
                "--bus",
                "inmemory",
                "--locus",
                "in-process",
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "task class:" in result.stderr
        assert not (_only_run_dir(tmp_path) / "diagnostics.log").exists()

    def test_the_streams_are_restored_afterwards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)
        before = (sys.stdout, sys.stderr)
        _invoke(tmp_path)
        assert (sys.stdout, sys.stderr) == before


class TestFailureIsOneLine:
    def test_a_terminal_failure_names_its_cause_and_the_receipt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path, failed=True)
        result = _invoke(tmp_path)
        match = _assert_single_error_line(result, tmp_path)
        receipt = json.loads(Path(match["path"]).read_text(encoding="utf-8"))
        assert match["cause"] == receipt["terminal_failure_cause"]
        assert match["message"]
        log = (_only_run_dir(tmp_path) / "diagnostics.log").read_text()
        assert f"{_DIAGNOSTIC_NOISE}: identity line" in log

    def test_the_two_exclusive_flags_are_refused_with_one_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)
        for flag in ("--json", "--human"):
            for stale in (tmp_path / "state").glob("runs/*"):
                for child in stale.iterdir():
                    child.unlink()
                stale.rmdir()
            result = _invoke(tmp_path, flag)
            match = _assert_single_error_line(result, tmp_path)
            assert match["cause"] == "refused_before_dispatch"
            assert "mutually exclusive" in match["message"]

    def test_an_argument_that_does_not_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)
        result = _invoke(tmp_path, "--task-type", "not-a-task-class")
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "argument_parsing"
        assert "not-a-task-class" in match["message"]
        assert "Usage:" not in result.stderr

    def test_an_unknown_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)
        result = _invoke(tmp_path, "--no-such-flag")
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "argument_parsing"

    def test_a_usage_error_raised_after_parsing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)
        result = _invoke(tmp_path, "--prompt-file", "-", prompt="a prompt")
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "refused_before_dispatch"

    def test_a_refusal_before_dispatch_from_a_check_that_names_the_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)

        def _refuse(**_kwargs: object) -> str:
            raise DelegateTenantRefusedError("no install identity: run onex login")

        monkeypatch.setattr(cli_delegate, "resolve_delegate_tenant", _refuse)
        result = _invoke(tmp_path)
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "refused_before_dispatch"
        assert "no install identity" in match["message"]
        log = (_only_run_dir(tmp_path) / "diagnostics.log").read_text()
        assert "task class:" in log
        assert "delegate artifacts (REFUSED" in log

    def test_an_unhandled_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)

        def _boom(**_kwargs: object) -> int:
            raise RuntimeError("the runtime fell over")

        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _boom)
        result = _invoke(tmp_path)
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "unhandled_error"
        assert "the runtime fell over" in match["message"]

    def test_a_multiline_cause_is_folded_to_one_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)

        def _boom(**_kwargs: object) -> int:
            raise RuntimeError("first line\nsecond line\n\nthird line")

        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _boom)
        result = _invoke(tmp_path)
        match = _assert_single_error_line(result, tmp_path)
        assert "first line" in match["message"]
        assert "third line" in match["message"]

    def test_an_exit_with_no_receipt_of_its_own(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", lambda **_k: 3)
        result = _invoke(tmp_path)
        assert result.exit_code == 3
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "failed"

    def test_the_handler_budget_timeout_is_one_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)

        def _hang(**_kwargs: object) -> int:
            raise cli_delegate.DelegateTimeoutExceededError(
                "onex delegate: no result within 240 s; the hard timeout fired."
            )

        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _hang)
        result = _invoke(tmp_path)
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "timeout"
        assert "Queue:" not in result.stderr

    def test_a_credential_in_the_cause_is_redacted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path)

        def _boom(**_kwargs: object) -> int:
            raise RuntimeError("bad call with api_key=sk-secret-123 rejected")

        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _boom)
        result = _invoke(tmp_path)
        _assert_single_error_line(result, tmp_path)
        assert "sk-secret-123" not in result.stderr

    def test_an_empty_answer_is_a_failure_not_a_silent_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _noisy_runtime(monkeypatch, tmp_path, response="")
        result = _invoke(tmp_path)
        match = _assert_single_error_line(result, tmp_path)
        assert match["cause"] == "empty_response"


class TestHelpAndFlagSurface:
    def test_help_lists_the_flag(self) -> None:
        result = CliRunner().invoke(delegate_command, ["--help"])
        assert result.exit_code == 0
        assert "--result-only" in result.stdout

    def test_without_the_flag_a_usage_error_keeps_clicks_own_form(
        self, tmp_path: Path
    ) -> None:
        result = CliRunner().invoke(
            delegate_command,
            ["x", "--task-type", "not-a-task-class", "--state-root", str(tmp_path)],
        )
        assert result.exit_code == 2
        assert "Usage:" in result.stderr

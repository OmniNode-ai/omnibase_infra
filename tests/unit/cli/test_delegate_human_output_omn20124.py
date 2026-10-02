# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Default ``onex delegate`` output is for a person (OMN-20124).

A person running ``onex delegate "say hello"`` gets the answer on stdout and a
one-line receipt summary on stderr. A failure is one plain line naming the
cause, the reason and the run id, and stdout stays empty. The full receipt JSON
is behind ``--json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.cli.delegate_human_output import render_delegate_outcome

pytestmark = pytest.mark.unit

RUN_ID = "11111111-2222-3333-4444-555555555555"


def _terminal(**overrides: object) -> dict[str, object]:
    terminal: dict[str, object] = {
        "attempts": [
            {
                "tier": "local",
                "backend_id": "local-qwen",
                "model_id": "qwen3-14b",
                "acceptance_decision": "accept",
                "quality_gate_passed": True,
            }
        ],
        "response": "Hello.",
        "model_name": "qwen3-14b",
        "provider": "http://localhost:8000",
        "metrics": {"cost_usd": 0.0012},
    }
    terminal.update(overrides)
    return terminal


def _envelope(
    terminal: dict[str, object], *, status: str = "success", exit_code: int = 0
) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "correlation_id": "99999999-2222-3333-4444-555555555555",
        "status": status,
        "exit_code": exit_code,
        "result_model": "omnimarket.ModelDelegateSkillResponse",
        "result": terminal,
    }


def _summary_envelope(
    *, error: str, error_type: str = "", terminal: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "correlation_id": "99999999-2222-3333-4444-555555555555",
        "status": "failed",
        "exit_code": 1,
        "result_model": "omnibase_infra.cli.ModelReceiptRuntimeSummary",
        "result": {
            "workflow": "/x/node_delegate_skill_orchestrator/contract.yaml",
            "workflow_result": "failed",
            "exit_code": 1,
            "terminal_payload": terminal,
            "handler_result": None,
            "error": error,
            "runtime_error_type": error_type,
            "wire_correlation_id": "abc",
        },
    }


def test_success_prints_the_answer_and_a_receipt_line(tmp_path: Path) -> None:
    outcome = render_delegate_outcome(_envelope(_terminal()), state_root=tmp_path)
    assert outcome is not None
    assert outcome.succeeded is True
    assert outcome.stdout == "Hello."
    assert len(outcome.stderr) == 1
    line = outcome.stderr[0]
    assert "qwen3-14b" in line
    assert "$0.0012" in line
    assert RUN_ID in line
    assert str(tmp_path.resolve() / "runs" / RUN_ID / "receipt.json") in line
    assert "{" not in outcome.stdout


def test_success_without_a_cost_says_so(tmp_path: Path) -> None:
    outcome = render_delegate_outcome(
        _envelope(_terminal(metrics=None)), state_root=tmp_path
    )
    assert outcome is not None
    assert "cost n/a" in outcome.stderr[0]


def test_failed_terminal_is_one_plain_line_with_cause_reason_and_run_id(
    tmp_path: Path,
) -> None:
    terminal = _terminal(
        attempts=[],
        response="",
        status="failed",
        terminal_failure_cause="no_provider_key",
        terminal_failure_reason="no model is declared and no provider key is set",
        error_message="no provider key",
    )
    outcome = render_delegate_outcome(
        _summary_envelope(error="", terminal=terminal), state_root=tmp_path
    )
    assert outcome is not None
    assert outcome.succeeded is False
    assert outcome.stdout == ""
    assert len(outcome.stderr) == 1
    line = outcome.stderr[0]
    assert "\n" not in line
    assert "no_provider_key" in line
    assert "no model is declared and no provider key is set" in line
    assert RUN_ID in line
    assert not line.lstrip().startswith("{")


def test_failure_falls_back_to_error_message_then_status(tmp_path: Path) -> None:
    terminal = _terminal(
        attempts=[], response="", error_message="every backend refused"
    )
    outcome = render_delegate_outcome(
        _summary_envelope(error="", terminal=terminal), state_root=tmp_path
    )
    assert outcome is not None
    assert "every backend refused" in outcome.stderr[0]


def test_runtime_error_traceback_collapses_to_its_last_line(tmp_path: Path) -> None:
    outcome = render_delegate_outcome(
        _summary_envelope(
            error="Traceback (most recent call last):\n  File x\nValueError: bad thing",
            error_type="ValueError",
        ),
        state_root=tmp_path,
    )
    assert outcome is not None
    assert outcome.succeeded is False
    assert len(outcome.stderr) == 1
    assert "ValueError: bad thing" in outcome.stderr[0]
    assert "Traceback" not in outcome.stderr[0]
    assert RUN_ID in outcome.stderr[0]


def test_non_delegation_receipt_is_not_rendered(tmp_path: Path) -> None:
    envelope = {
        "run_id": RUN_ID,
        "status": "success",
        "exit_code": 0,
        "result_model": "some.other.Model",
        "result": {"x": 1},
    }
    assert render_delegate_outcome(envelope, state_root=tmp_path) is None


def test_json_is_never_produced_by_the_human_renderer(tmp_path: Path) -> None:
    outcome = render_delegate_outcome(_envelope(_terminal()), state_root=tmp_path)
    assert outcome is not None
    with pytest.raises(json.JSONDecodeError):
        json.loads(outcome.stdout)


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "delegation"


def _recorded(name: str) -> dict[str, object]:
    loaded = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_recorded_failed_run_renders_one_plain_line_with_the_run_id(
    tmp_path: Path,
) -> None:
    envelope = _recorded("omn18306/failed_no_accepted_attempt_receipt.json")
    outcome = render_delegate_outcome(envelope, state_root=tmp_path)
    assert outcome is not None
    assert outcome.succeeded is False
    assert outcome.stdout == ""
    assert len(outcome.stderr) == 1
    line = outcome.stderr[0]
    assert str(envelope["run_id"]) in line
    assert "\n" not in line
    assert line.startswith("onex delegate failed: ")


def test_recorded_dispatched_run_renders_the_answer(tmp_path: Path) -> None:
    envelope = _recorded("omn18569/dispatched_envelope_carrier_receipt.json")
    outcome = render_delegate_outcome(envelope, state_root=tmp_path)
    assert outcome is not None
    if outcome.succeeded:
        assert outcome.stdout
        assert not outcome.stdout.lstrip().startswith("{")
    else:
        assert outcome.stdout == ""
    assert len(outcome.stderr) == 1
    assert str(envelope["run_id"]) in outcome.stderr[0]


def test_the_cli_renderer_prints_answer_to_stdout_and_summary_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from omnibase_infra.cli import cli_delegate

    receipt = _FakeReceipt(_envelope(_terminal()))
    succeeded = cli_delegate._render_receipt_for_person(receipt, state_root=tmp_path)
    captured = capsys.readouterr()
    assert succeeded is True
    assert captured.out == "Hello.\n"
    assert "qwen3-14b" in captured.err
    assert RUN_ID in captured.err


def test_the_cli_renderer_prints_json_for_a_receipt_that_is_not_a_delegation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from omnibase_infra.cli import cli_delegate

    receipt = _FakeReceipt(
        {"run_id": RUN_ID, "result_model": "some.other.Model", "result": {"x": 1}}
    )
    cli_delegate._render_receipt_for_person(receipt, state_root=tmp_path)
    assert json.loads(capsys.readouterr().out) == receipt.envelope


class _FakeReceipt:
    def __init__(self, envelope: dict[str, object]) -> None:
        self.envelope = envelope

    def model_dump(self, *, mode: str) -> dict[str, object]:
        return self.envelope

    def model_dump_json(self) -> str:
        return json.dumps(self.envelope)


def test_a_reaper_no_terminal_terminal_renders_as_one_plain_failure_line(
    tmp_path: Path,
) -> None:
    """OMN-19441: the delegation reaper's terminal decodes in the CLI renderer.

    The reaper closes a command whose worker never answered with cause
    ``no_terminal``. The renderer must treat it like any other failed terminal:
    stdout stays empty and the cause, the reason and the run id are on one line.
    """
    terminal = _terminal(
        attempts=[],
        response="",
        status="failed",
        terminal_failure_cause="no_terminal",
        error_message=(
            "the command was claimed and produced no terminal by its deadline; "
            "the delegation reaper closed it"
        ),
    )
    outcome = render_delegate_outcome(
        _summary_envelope(error="", terminal=terminal), state_root=tmp_path
    )
    assert outcome is not None
    assert outcome.succeeded is False
    assert outcome.stdout == ""
    assert len(outcome.stderr) == 1
    line = outcome.stderr[0]
    assert "\n" not in line
    assert "no_terminal" in line
    assert "the delegation reaper closed it" in line
    assert RUN_ID in line

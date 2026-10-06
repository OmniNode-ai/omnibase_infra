# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Cover delegation carrier fallback and human failure-message precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.cli import delegate_human_output

_RUN_ID = "11111111-2222-3333-4444-555555555555"
_WORKFLOW = "node_delegate_skill_orchestrator"


def _summary(**fields: object) -> dict[str, object]:
    return {
        "run_id": _RUN_ID,
        "status": "failed",
        "exit_code": 1,
        "result_model": "ModelReceiptRuntimeSummary",
        "result": {"workflow": _WORKFLOW, "wire_correlation_id": "published", **fields},
    }


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("", 400, ""),
        (" \n\t\n ", 400, ""),
        (" first \n\n second ", 400, "first second"),
        (
            "Traceback (most recent call last):\n frame\nValueError: bad",
            400,
            "ValueError: bad",
        ),
        ("12345678", 8, "12345678"),
        ("123456789", 8, "12345..."),
    ],
)
def test_one_line_handles_blank_traceback_and_length_boundaries(
    text: str, limit: int, expected: str
) -> None:
    assert delegate_human_output.one_line(text, limit=limit) == expected


@pytest.mark.parametrize("result", [None, [], "not a mapping"])
def test_non_mapping_result_is_not_a_delegation(tmp_path: Path, result: object) -> None:
    envelope = {"result_model": "ModelDelegateSkillResponse", "result": result}
    assert (
        delegate_human_output.render_delegate_outcome(envelope, state_root=tmp_path)
        is None
    )
    assert delegate_human_output._terminal_of(envelope) is None


def test_invalid_primary_carrier_falls_back_to_handler_result(tmp_path: Path) -> None:
    terminal = {
        "attempts": [{"acceptance_decision": "accept", "model_id": "local-model"}],
        "response": "Recovered answer.",
    }
    envelope = _summary(
        terminal_payload={"response": "missing attempts"}, handler_result=terminal
    )
    envelope["exit_code"] = 0

    outcome = delegate_human_output.render_delegate_outcome(
        envelope, state_root=tmp_path
    )

    assert outcome is not None
    assert outcome.succeeded
    assert outcome.stdout == "Recovered answer."
    assert "model local-model, cost n/a" in outcome.stderr[0]


def test_pre_publish_failure_names_cause_without_implicating_deployed_lane(
    tmp_path: Path,
) -> None:
    envelope = _summary(
        wire_correlation_id="",
        workflow_result="failed",
        runtime_error_type="ValidationError",
    )
    outcome = delegate_human_output.render_delegate_outcome(
        envelope, state_root=tmp_path
    )

    assert outcome is not None
    assert not outcome.succeeded
    assert outcome.stdout == ""
    assert outcome.stderr[0].startswith(
        f"onex delegate failed: delegate run {_RUN_ID} failed before publish "
        "(ValidationError):"
    )
    assert "no command reached the broker" in outcome.stderr[0]
    assert "The deployed lane is not implicated." in outcome.stderr[0]
    assert "\n" not in outcome.stderr[0]


@pytest.mark.parametrize(
    ("terminal_fields", "expected"),
    [
        (
            {
                "terminal_failure_cause": " queue_timeout ",
                "terminal_failure_reason": " \n ",
            },
            "queue_timeout",
        ),
        ({"terminal_failure_reason": " reason only "}, "reason only"),
        (
            {
                "attempts": [
                    {"error_message": "first failure"},
                    {"error_message": ""},
                    {"error_message": "second failure"},
                ]
            },
            "first failure second failure",
        ),
        (
            {"terminal_failure_reason": "preferred", "error_message": "secondary"},
            "preferred; secondary",
        ),
        (
            {"terminal_failure_cause": " \n ", "error_message": "fallback error"},
            "fallback error",
        ),
    ],
)
def test_terminal_failure_preserves_reason_and_error(
    tmp_path: Path, terminal_fields: dict[str, object], expected: str
) -> None:
    envelope = _summary(
        terminal_payload={"attempts": [], **terminal_fields}, error="runtime fallback"
    )
    outcome = delegate_human_output.render_delegate_outcome(
        envelope, state_root=tmp_path
    )

    assert outcome is not None
    assert not outcome.succeeded
    assert outcome.stdout == ""
    assert outcome.stderr[0].startswith(
        f"onex delegate failed: {expected} (run {_RUN_ID};"
    )
    assert "runtime fallback" not in outcome.stderr[0]


@pytest.mark.parametrize(
    ("error_type", "error", "expected"),
    [
        ("ValueError", "bad input", "ValueError: bad input"),
        ("ValueError", "ValueError: bad input", "ValueError: bad input"),
        ("ValueError", "", "ValueError"),
        ("", "bad input", "bad input"),
        ("", "", "the run ended with status failed"),
    ],
)
def test_runtime_failure_fallbacks_when_terminal_has_no_reason(
    tmp_path: Path, error_type: str, error: str, expected: str
) -> None:
    envelope = _summary(
        terminal_payload={"attempts": []}, runtime_error_type=error_type, error=error
    )
    outcome = delegate_human_output.render_delegate_outcome(
        envelope, state_root=tmp_path
    )

    assert outcome is not None
    assert not outcome.succeeded
    assert outcome.stdout == ""
    assert outcome.stderr[0].startswith(
        f"onex delegate failed: {expected} (run {_RUN_ID};"
    )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: CI Summary requires the repo-owned evidence verdict on PR runs.

Scratch control: this docstring line changes no behaviour, so the test passes at
the merge base and at the head alike.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_SWEEP_EXCLUSIONS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    check_run_event_index,
    evaluate_external_contexts,
    evaluate_external_sweep,
    main,
)

pytestmark = pytest.mark.unit

_DOD_VERIFY = "repo-evidence / dod-verify"
_RUN_ID = 37928489908
_NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)
_RUNS: list[dict[str, object]] = [
    {"id": 1, "event": "pull_request"},
    {"id": _RUN_ID, "event": "pull_request_target"},
]


def _row(name: str, conclusion: str = "success") -> dict[str, object]:
    run_id = _RUN_ID if name == _DOD_VERIFY else 1
    return {
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "started_at": "2026-10-08T23:00:00Z",
        "completed_at": "2026-10-08T23:05:00Z",
        "html_url": (
            f"https://github.com/OmniNode-ai/omnibase_infra/actions/runs/{run_id}/job/1"
        ),
    }


def _other_expected_rows() -> list[dict[str, object]]:
    return [_row(c) for c in EXPECTED_EXTERNAL_CONTEXTS if c != _DOD_VERIFY]


def _pr_summary(
    rows: list[dict[str, object]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str]:
    """Exercise the real CLI's pull_request event scoping with green CI jobs."""
    jobs_file = tmp_path / "jobs.json"
    checks_file = tmp_path / "checks.json"
    runs_file = tmp_path / "runs.json"
    jobs_file.write_text(
        json.dumps([_row(c) for c in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)]),
        encoding="utf-8",
    )
    checks_file.write_text(json.dumps(rows), encoding="utf-8")
    runs_file.write_text(json.dumps(_RUNS), encoding="utf-8")
    code = main(
        [
            "--event-name",
            "pull_request",
            "--jobs-file",
            str(jobs_file),
            "--check-runs-file",
            str(checks_file),
            "--workflow-runs-file",
            str(runs_file),
        ]
    )
    return code, capsys.readouterr().out


def test_dod_verify_is_an_expected_external_context() -> None:
    assert _DOD_VERIFY in EXPECTED_EXTERNAL_CONTEXTS
    assert "repo-evidence / verify" not in EXPECTED_EXTERNAL_CONTEXTS


def test_absent_dod_verify_blocks_pull_request_ci_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _other_expected_rows()
    failures, unresolved = evaluate_external_contexts(rows, EXPECTED_EXTERNAL_CONTEXTS)
    assert failures == []
    assert unresolved == [_DOD_VERIFY]
    code, report = _pr_summary(rows, tmp_path, capsys)
    # Absence is PENDING while polling, converted to FAILURE at CI's deadline.
    # It must never produce a successful required CI Summary verdict.
    assert code == EXIT_PENDING, report
    assert _DOD_VERIFY in report


def test_green_pull_request_target_dod_verify_satisfies_ci_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [*_other_expected_rows(), _row(_DOD_VERIFY)]
    assert evaluate_external_contexts(rows, EXPECTED_EXTERNAL_CONTEXTS) == ([], [])
    failures, in_flight, _swept, excluded, provisional = evaluate_external_sweep(
        rows,
        expected=EXPECTED_EXTERNAL_CONTEXTS,
        in_run_names=frozenset(),
        self_name="CI Summary",
        exclusions=EXTERNAL_SWEEP_EXCLUSIONS,
        events=check_run_event_index(_RUNS),
        now=_NOW,
    )
    assert (failures, in_flight, excluded, provisional) == ([], [], [], [])
    code, report = _pr_summary(rows, tmp_path, capsys)
    assert code == EXIT_SUCCESS, report


@pytest.mark.parametrize("conclusion", ["failure", "skipped"])
def test_refused_dod_verify_fails_ci_summary(
    conclusion: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [*_other_expected_rows(), _row(_DOD_VERIFY, conclusion)]
    failures, unresolved = evaluate_external_contexts(rows, EXPECTED_EXTERNAL_CONTEXTS)
    assert failures == [_DOD_VERIFY]
    assert unresolved == []
    code, report = _pr_summary(rows, tmp_path, capsys)
    assert code == EXIT_FAILURE, report
    assert _DOD_VERIFY in report

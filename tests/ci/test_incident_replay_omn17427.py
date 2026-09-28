# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for ``scripts/ci/ci_summary_gate.py`` (OMN-17427).

THE INCIDENT
    ``CI Summary`` run 36336140897 attempt 1 on ``omnibase_infra#4216`` (head
    ``09f3839a``) exited FAILURE at 17:28:35Z on 2026-09-27 with:

    * ``external-context failures: call-reject-skip-token / scan /
      reject-skip-gate-token``
    * ``external sweep failures ...: Hostile Review Gate
      (cancelled_without_replacement: cancelled 636s ago, past the 600s re-run
      grace, and no replacement row exists on this head ...)`` and the same
      for ``Hostile Review Thread Gate``.

    Both producers had been cancelled at 17:17:59Z by NEWER runs of the same
    workflows for the same head, created at 17:17:43Z (Hostile Reviewer run
    36336398239, Reject skip-gate bypass tokens run 36336398785). Those runs
    were queued behind the runner fleet and wrote their rows at 17:47Z, all
    ``success``. Attempt 2 of the same CI run passed with no change to the
    head. The fixed 600s grace guessed that a replacement was coming; the
    workflow-runs payload the poller already fetched said so outright, and for
    as long as it was true. The same shape reddened #4209, #4218 and #4219 the
    same day at 631-649s.

THE CAPTURES
    ``tests/fixtures/omn17427/`` holds the head's check-runs, its workflow
    runs and the CI run's attempt-1 jobs, fetched live after the fact and cut
    back to the verdict instant: a row or job that had not started by 17:28:35Z
    is dropped, one that finished later is shown in progress, and a workflow
    run updated after the instant is shown in progress. The cut reproduces
    the poller's own log line ``jobs observed: 22``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXPECTED_EXTERNAL_CONTEXTS,
    evaluate,
    replacement_run_in_flight,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "omn17427"
CHECK_RUNS = FIXTURES / "infra-pr4216-head-09f3839a-check-runs-at-172835.json.captured"
WORKFLOW_RUNS = (
    FIXTURES / "infra-pr4216-head-09f3839a-workflow-runs-at-172835.json.captured"
)
JOBS = FIXTURES / "infra-pr4216-ci-run-36336140897-a1-jobs-at-172835.json.captured"

INSTANT = datetime.fromisoformat("2026-09-27T17:28:35+00:00")
CI_RUN_ID = 36336140897
REPLACEMENT_RUN_IDS = {36336398239, 36336398785}


def _load(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def _evaluate(workflow_runs: list[dict[str, object]]) -> tuple[int, str]:
    return evaluate(
        _load(JOBS),
        run_attempt=1,
        check_runs=_load(CHECK_RUNS),
        external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        workflow_runs=workflow_runs,
        current_run_id=CI_RUN_ID,
        now=INSTANT,
    )


def test_the_capture_reproduces_the_poller_snapshot() -> None:
    assert len(_load(JOBS)) == 22


def test_a_replacement_in_flight_holds_pending_instead_of_failing() -> None:
    code, report = _evaluate(_load(WORKFLOW_RUNS))
    assert code == EXIT_PENDING, report
    assert "cancelled_without_replacement" not in report
    assert "external-context failures" not in report


def test_without_the_replacement_the_same_rows_still_fail_closed() -> None:
    """Falsifier and fail-closed control: had the newer runs finished without
    writing their rows, the cancellations are the head's answer and fail."""
    finished = [
        {**r, "status": "completed", "conclusion": "success"}
        if r["id"] in REPLACEMENT_RUN_IDS
        else r
        for r in _load(WORKFLOW_RUNS)
    ]
    code, report = _evaluate(finished)
    assert code == EXIT_FAILURE, report
    assert "Hostile Review Gate (cancelled_without_replacement" in report
    assert "call-reject-skip-token / scan / reject-skip-gate-token" in report


def test_no_workflow_runs_payload_is_the_strict_reading() -> None:
    """A failed workflow-runs fetch leaves the file absent: nothing is held."""
    code, report = _evaluate([])
    assert code == EXIT_FAILURE, report


# --- replacement_run_in_flight, shape by shape ------------------------------


def _row(run_id: int, completed_at: str = "2026-09-27T17:17:59Z") -> dict[str, object]:
    return {
        "name": "Some Gate",
        "status": "completed",
        "conclusion": "cancelled",
        "completed_at": completed_at,
        "html_url": f"https://github.com/o/r/actions/runs/{run_id}/job/1",
    }


def _run(
    run_id: int,
    *,
    status: str = "completed",
    workflow_id: int = 7,
    event: str = "pull_request",
    run_started_at: str = "2026-09-27T17:10:00Z",
) -> dict[str, object]:
    return {
        "id": run_id,
        "workflow_id": workflow_id,
        "event": event,
        "status": status,
        "run_started_at": run_started_at,
    }


def test_a_newer_run_of_the_same_workflow_still_running_counts() -> None:
    runs = [_run(100), _run(200, status="queued")]
    assert replacement_run_in_flight(_row(100), runs)


def test_a_newer_run_that_finished_does_not_count() -> None:
    runs = [_run(100), _run(200)]
    assert not replacement_run_in_flight(_row(100), runs)


def test_an_older_run_still_running_does_not_count() -> None:
    runs = [_run(50, status="in_progress"), _run(100)]
    assert not replacement_run_in_flight(_row(100), runs)


def test_another_workflow_does_not_count() -> None:
    runs = [_run(100), _run(200, status="queued", workflow_id=8)]
    assert not replacement_run_in_flight(_row(100), runs)


def test_another_event_does_not_count() -> None:
    runs = [_run(100), _run(200, status="queued", event="push")]
    assert not replacement_run_in_flight(_row(100), runs)


def test_a_rerun_attempt_of_the_same_run_counts() -> None:
    runs = [_run(100, status="in_progress", run_started_at="2026-09-27T17:30:00Z")]
    assert replacement_run_in_flight(_row(100), runs)


def test_a_sibling_cancelled_inside_the_running_attempt_does_not_count() -> None:
    """A fail-fast matrix cancels a sibling while its own attempt still runs;
    that cancellation is this attempt's answer, not a superseded one."""
    runs = [_run(100, status="in_progress", run_started_at="2026-09-27T17:10:00Z")]
    assert not replacement_run_in_flight(_row(100), runs)


def test_a_row_with_no_run_url_does_not_count() -> None:
    row = _row(100)
    del row["html_url"]
    assert not replacement_run_in_flight(row, [_run(100), _run(200, status="queued")])


def test_a_run_missing_from_the_payload_does_not_count() -> None:
    assert not replacement_run_in_flight(_row(100), [_run(200, status="queued")])


# --- the run's own rows are never swept -------------------------------------


def test_a_row_this_run_wrote_is_left_to_the_in_run_layers() -> None:
    """A job created straight to ``skipped`` after the jobs fetch is already on
    the head when the check-runs fetch runs. Measured on onex_change_control
    CI run 36331600130 (``Migration Inventory Sync (skipped)`` swept as a red
    nothing names); this module had the same ``in_run_names`` shape."""
    own = {
        "id": 1,
        "name": "A Job Listed After The Jobs Fetch",
        "status": "completed",
        "conclusion": "skipped",
        "started_at": "2026-09-27T17:28:30Z",
        "completed_at": "2026-09-27T17:28:30Z",
        "html_url": (
            f"https://github.com/OmniNode-ai/omnibase_infra/actions/runs/{CI_RUN_ID}/job/1"
        ),
    }
    code, report = evaluate(
        _load(JOBS),
        run_attempt=1,
        check_runs=[*_load(CHECK_RUNS), own],
        external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        workflow_runs=_load(WORKFLOW_RUNS),
        current_run_id=CI_RUN_ID,
        now=INSTANT,
    )
    assert code == EXIT_PENDING, report
    assert "A Job Listed After The Jobs Fetch" not in report.split("external sweep")[-1]

    # The same row from another run, settled past every grace, still reds.
    foreign = {
        **own,
        "started_at": "2026-09-27T16:00:00Z",
        "completed_at": "2026-09-27T16:00:00Z",
        "html_url": "https://github.com/o/r/actions/runs/1/job/1",
    }
    code, report = evaluate(
        _load(JOBS),
        run_attempt=1,
        check_runs=[*_load(CHECK_RUNS), foreign],
        external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        workflow_runs=_load(WORKFLOW_RUNS),
        current_run_id=CI_RUN_ID,
        now=INSTANT,
    )
    assert code == EXIT_FAILURE, report
    assert "A Job Listed After The Jobs Fetch (skipped)" in report


def test_the_poller_passes_its_own_run_id() -> None:
    text = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    assert '--current-run-id "${RUN_ID}"' in text

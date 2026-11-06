# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The test matrix starts when the quick gates end, not when Lint ends (OMN-22545).

``detect-changes`` and ``test-parallel`` used to list ``lint`` in ``needs``. Over
the 68 completed ``ci.yml`` runs read on 2026-11-06, Lint ended at a median of
9.4 minutes and the first test shard started at a median of 10.4 minutes, while
the fourteen other gates ``detect-changes`` waits on ended at a median of 3.0
minutes. OMN-19613 recorded the rule for this change: keep lint in the tests'
``needs`` and drop it if Lint still ends more than 60 seconds after ONEX
Validators. Lint ended about five minutes after it.

What this file pins, and what it does not move:

* neither job lists ``lint`` again (a ``needs`` edge back to it puts the test
  start behind Lint's 9 minutes);
* ``Lint`` is still a STRICT gate of ``CI Summary``, and ``CI Summary`` is still
  the no-``needs`` poller, so a red Lint still fails the required context
  (tests/ci/test_ci_summary_gate.py::TestCiSummaryGate::
  test_lint_failure_alone_fails_ci_summary runs the verdict itself).

The control builds the pre-change shape (``lint`` added back to a copy of the
live workflow) and shows the check names the offender, so a vacuous pass cannot
hide behind an empty list.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_SUCCESS,
    STRICT_GATE_JOBS,
    evaluate,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
LINT_JOB_ID = "lint"
TEST_PATH_JOBS = ("detect-changes", "test-parallel")


def _load() -> dict[Any, Any]:
    with CI_WORKFLOW.open(encoding="utf-8") as handle:
        loaded: dict[Any, Any] = yaml.safe_load(handle)
    return loaded


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def jobs_waiting_on_lint(workflow: dict[Any, Any]) -> list[str]:
    """Names of the test-path jobs that still list ``lint`` in ``needs``."""
    jobs = workflow["jobs"]
    return [name for name in TEST_PATH_JOBS if LINT_JOB_ID in _needs(jobs[name])]


@pytest.mark.parametrize("job_name", TEST_PATH_JOBS)
def test_test_path_job_does_not_wait_for_lint(job_name: str) -> None:
    needs = _needs(_load()["jobs"][job_name])
    assert needs, f"{job_name} lost every dependency; this test's premise is gone"
    assert LINT_JOB_ID not in needs, (
        f"{job_name} lists lint in needs again, so the test shards start after "
        f"Lint's median 9.4 minutes instead of the quick gates' 3.0. Got: {needs!r}"
    )


def test_the_other_quick_gates_still_gate_the_test_matrix() -> None:
    """Only lint left the list: the other fourteen gates and the zone filter stay."""
    jobs = _load()["jobs"]
    for name in ("onex-validation", "schema-handshake", "contract-sync-gate"):
        assert name in _needs(jobs["detect-changes"])
        assert name in _needs(jobs["test-parallel"])
    assert "zone-filter" in _needs(jobs["test-parallel"])
    assert "detect-changes" in _needs(jobs["test-parallel"])


def test_lint_is_still_a_strict_gate_of_ci_summary() -> None:
    jobs = _load()["jobs"]
    lint_name = jobs[LINT_JOB_ID]["name"]
    assert lint_name in STRICT_GATE_JOBS
    assert "needs" not in jobs["ci-summary"], (
        "CI Summary is the no-needs poller (OMN-14127); a needs edge would let "
        "an absent Lint check-run leave the required context absent"
    )


def test_lint_job_still_runs_on_every_event() -> None:
    """A strict gate that is skipped fails closed, so Lint must not gain an ``if:``."""
    assert "if" not in _load()["jobs"][LINT_JOB_ID]


def test_control_the_check_names_the_offender_when_lint_is_added_back() -> None:
    """Injected break (named): put ``lint`` back into both needs lists of a copy."""
    live = _load()
    assert jobs_waiting_on_lint(live) == []
    broken = copy.deepcopy(live)
    for name in TEST_PATH_JOBS:
        broken["jobs"][name]["needs"] = [LINT_JOB_ID, *_needs(broken["jobs"][name])]
    assert jobs_waiting_on_lint(broken) == list(TEST_PATH_JOBS)
    one = copy.deepcopy(live)
    one["jobs"]["test-parallel"]["needs"] = [
        LINT_JOB_ID,
        *_needs(one["jobs"]["test-parallel"]),
    ]
    assert jobs_waiting_on_lint(one) == ["test-parallel"]


# --------------------------------------------------------------------------- #
# replay of a recorded run
# --------------------------------------------------------------------------- #

TESTS_START_BUDGET_MIN = 6.0
LINT_GATED_START_FLOOR_MIN = 10.0


def _minutes(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _recorded_job(jobs: list[dict[str, Any]], job_id: str, name: str | None) -> Any:
    """The recorded job for a workflow job id.

    Matches a ``name:``, or the ``<id> / <job>`` label a reusable workflow gets.
    """
    for recorded in jobs:
        label = recorded["name"]
        if (
            label == name
            or label.startswith(f"{job_id} / ")
            or label.endswith(f" / {job_id}")
        ):
            return recorded
    raise AssertionError(f"no recorded job for {job_id!r} ({name!r})")


def earliest_tests_start_minutes(
    workflow: dict[Any, Any], recorded: list[dict[str, Any]]
) -> float:
    """When ``test-parallel`` could first start if only its ``needs`` held it back.

    Each job that has dependencies starts when the last of them ends, and takes
    as long as it took in the recorded run. A job with none starts when it did
    in the recorded run. Offsets are minutes from the recorded run's first job.
    """
    jobs = workflow["jobs"]
    origin = min(_minutes(r["started_at"]) for r in recorded if r["started_at"])
    memo: dict[str, float] = {}

    def end(job_id: str) -> float:
        if job_id in memo:
            return memo[job_id]
        job = jobs[job_id]
        row = _recorded_job(recorded, job_id, job.get("name"))
        began = _minutes(row["started_at"])
        took = (_minutes(row["completed_at"]) - began).total_seconds() / 60
        needs = _needs(job)
        start = (
            max(end(n) for n in needs)
            if needs
            else (began - origin).total_seconds() / 60
        )
        memo[job_id] = start + took
        return memo[job_id]

    needs = _needs(jobs["test-parallel"])
    return max(end(n) for n in needs)


@pytest.mark.live_contact("tests/ci/fixtures/ci_run_42139619077_jobs_omn22545.json")
def test_recorded_run_would_start_its_tests_before_lint_ends(
    recorded_response: dict[str, object],
) -> None:
    """Replay of a real run: Lint ended at 11.5 min and the first shard started at 13.1.

    Under the live ``needs`` the tests could have started once the quick gates
    and Detect Changes had run. With ``lint`` added back (the control) they cannot
    start before Lint ends.
    """
    recorded = recorded_response["jobs"]
    assert isinstance(recorded, list)
    origin = min(_minutes(r["started_at"]) for r in recorded if r["started_at"])
    lint = next(r for r in recorded if r["name"] == "Lint")
    lint_end = (_minutes(lint["completed_at"]) - origin).total_seconds() / 60
    first_shard = min(
        _minutes(r["started_at"])
        for r in recorded
        if r["name"].startswith("Tests (Split") and r["started_at"]
    )
    shard_start = (first_shard - origin).total_seconds() / 60
    assert lint_end > LINT_GATED_START_FLOOR_MIN
    assert shard_start > lint_end, "the recording was taken under the old needs"

    live = _load()
    assert earliest_tests_start_minutes(live, recorded) < TESTS_START_BUDGET_MIN

    broken = copy.deepcopy(live)
    broken["jobs"]["detect-changes"]["needs"] = [
        LINT_JOB_ID,
        *_needs(broken["jobs"]["detect-changes"]),
    ]
    assert earliest_tests_start_minutes(broken, recorded) > LINT_GATED_START_FLOOR_MIN


@pytest.mark.live_contact("tests/ci/fixtures/ci_run_42139619077_jobs_omn22545.json")
def test_recorded_run_flips_ci_summary_red_when_only_lint_is_red(
    recorded_response: dict[str, object],
) -> None:
    """Named injected break: the recorded green run with its Lint row set to failure.

    The 90 jobs are the real run. Unchanged they give the SUCCESS verdict; with
    only ``Lint`` concluding ``failure`` the verdict is FAILURE and names Lint,
    so removing Lint from the test matrix's ``needs`` leaves the merge held.
    """
    recorded = recorded_response["jobs"]
    assert isinstance(recorded, list)
    jobs: list[dict[str, object]] = [
        {
            "name": r["name"],
            "status": "completed",
            "conclusion": r["conclusion"],
            "run_attempt": 1,
        }
        for r in recorded
    ]
    code, _ = evaluate(jobs)
    assert code == EXIT_SUCCESS
    red = [dict(j, conclusion="failure") if j["name"] == "Lint" else j for j in jobs]
    code, report = evaluate(red)
    assert code == EXIT_FAILURE
    assert "Lint: completed/failure" in report

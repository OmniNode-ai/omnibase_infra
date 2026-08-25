# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for scripts/ci/pr_ci_zombie_detector.py (OMN-15547 / OMN-16494).

Drives the real ``determine_zombie_cancellations`` decision function with facts
pulled from the captured, unmodified GitHub API bytes of the exact incident
this guard exists to catch: omnimarket#2106, run 32750189057 (old head)
wedging the shared `pull_request` concurrency group against run 32750203422
(new head), 2026-08-24T16:19-16:26Z.

Why the replay is built from job timestamps, not the runs' own `status`
field: the incident is over. The GitHub API only ever exposes CURRENT state,
so re-fetching either run today returns `status: completed` -- there is no
live endpoint that still says "queued" or "in_progress" for a wedge that was
force-cancelled on 2026-08-24. What GitHub does retain forever, unchanged, is
each JOB's own `started_at`/`completed_at` -- a job's history does not get
edited after the fact. Those captured job timestamps are what let this test
reconstruct, honestly and without inventing anything, that at 16:26:00Z (a
real instant strictly between the two proven facts below) the old run was
still non-terminal and the new run genuinely had zero jobs scheduled -- which
is exactly the state ``determine_zombie_cancellations`` must reject.

Also captures a second, independent finding from building this very replay:
the runs API's `pull_requests[]` field -- the field the detector's FIRST
implementation used to group runs -- came back an EMPTY array for BOTH of
these runs, despite neither being a fork PR (see the captured run-level
fixtures). That is why the shipped detector groups by `head_branch` instead;
see ``pr_ci_zombie_detector.py``'s module docstring and
``test_pr_ci_zombie_detector.py::test_empty_pull_requests_field_does_not_matter_grouping_is_by_head_branch``
for the sibling regression guard.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.ci.pr_ci_zombie_detector import (
    PullRequestRun,
    determine_zombie_cancellations,
)

pytestmark = pytest.mark.ci

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "omn16494"

_OLD_RUN_ID = 32750189057
_NEW_RUN_ID = 32750203422


def _load(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


class TestOmn16494IncidentReplay:
    def test_real_omnimarket_2106_wedge_is_rejected(self) -> None:
        old_run = _load("omnimarket-run-32750189057.gh-api.json.captured")
        new_run = _load("omnimarket-run-32750203422.gh-api.json.captured")
        old_jobs = _load("omnimarket-run-32750189057-jobs.gh-api.json.captured")
        new_jobs_attempt1 = _load(
            "omnimarket-run-32750203422-attempt1-jobs.gh-api.json.captured"
        )

        assert old_run["id"] == _OLD_RUN_ID
        assert new_run["id"] == _NEW_RUN_ID
        assert old_run["head_branch"] == new_run["head_branch"], (
            "sanity: this replay only makes sense if both runs share a head "
            "branch -- the exact key determine_zombie_cancellations groups on"
        )

        # Ground truth pulled directly from the captured, immutable job
        # timestamps -- not invented. A job's started_at/completed_at never
        # changes after the fact.
        old_run_last_job_completed_at = max(
            job["completed_at"] for job in old_jobs["jobs"]
        )
        new_run_first_job_started_at = min(
            job["started_at"] for job in new_jobs_attempt1["jobs"] if job["started_at"]
        )
        assert old_run_last_job_completed_at == "2026-08-24T16:26:25Z"
        assert new_run_first_job_started_at == "2026-08-24T16:26:29Z"

        # observed_at is a real instant strictly between "the old run's last
        # job finished" and "the new run's first job started" -- a moment a
        # scheduled detector could genuinely have observed this wedge live.
        observed_at = "2026-08-24T16:26:00Z"
        assert old_run["created_at"] < observed_at < old_run_last_job_completed_at
        assert new_run["created_at"] < observed_at < new_run_first_job_started_at

        runs = [
            PullRequestRun(
                repo="omnimarket",
                run_id=old_run["id"],
                head_branch=old_run["head_branch"],
                head_sha=old_run["head_sha"],
                # True as of observed_at: its last job does not complete
                # until 16:26:25Z, strictly after observed_at.
                status="in_progress",
                created_at=old_run["created_at"],
            ),
            PullRequestRun(
                repo="omnimarket",
                run_id=new_run["id"],
                head_branch=new_run["head_branch"],
                head_sha=new_run["head_sha"],
                # True as of observed_at: its first job does not start until
                # 16:26:29Z, strictly after observed_at -- zero jobs scheduled.
                status="queued",
                created_at=new_run["created_at"],
                job_count=0,
            ),
        ]

        candidates = determine_zombie_cancellations(
            runs=runs,
            # ~16 min after the new run's created_at (16:19:55Z) -- past the
            # 900s default staleness threshold, matching how long this
            # incident actually sat wedged before a human force-cancelled it.
            now=datetime(2026, 8, 24, 16, 36, 0, tzinfo=UTC),
            stale_after_seconds=900,
        )

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.run_id == _OLD_RUN_ID
        assert candidate.blocked_run_id == _NEW_RUN_ID
        assert candidate.reason == "stale_head_blocking_concurrency_group"
        assert candidate.head_branch == old_run["head_branch"]

    def test_empty_pull_requests_field_on_the_real_captured_runs(self) -> None:
        """The independent finding this replay surfaced: both real run
        objects carry an empty `pull_requests` array despite neither being a
        fork PR. This is the fact that made head_branch grouping mandatory,
        not optional -- documented here against the actual captured bytes so
        the claim in the module docstring stays checked, not just asserted
        in prose."""
        old_run = _load("omnimarket-run-32750189057.gh-api.json.captured")
        new_run = _load("omnimarket-run-32750203422.gh-api.json.captured")

        assert old_run["pull_requests"] == []
        assert new_run["pull_requests"] == []
        assert old_run["head_branch"] and new_run["head_branch"]

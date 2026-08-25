# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from datetime import UTC, datetime

from scripts.ci.pr_ci_zombie_detector import (
    PullRequestRun,
    determine_zombie_cancellations,
)

_NOW = datetime(2026, 8, 24, 17, 0, 0, tzinfo=UTC)
_BRANCH = "jonah/omn-15683-tenant-uuid-slug-migration"


def _run(
    run_id: int,
    *,
    repo: str = "omnimarket",
    head_branch: str = _BRANCH,
    status: str = "in_progress",
    head_sha: str = "1ac4fc40",
    created_at: str = "2026-08-24T16:19:46Z",
    job_count: int | None = None,
) -> PullRequestRun:
    return PullRequestRun(
        repo=repo,
        run_id=run_id,
        head_branch=head_branch,
        head_sha=head_sha,
        status=status,
        created_at=created_at,
        job_count=job_count,
    )


def test_flags_stale_head_run_blocking_a_wedged_zero_job_newer_run() -> None:
    """Reproduces the verified omnimarket#2106 / onex_change_control#7009 shape:
    an older-head run still active, a newer-head run queued with zero jobs
    for longer than the threshold."""
    candidates = determine_zombie_cancellations(
        runs=[
            _run(
                32750189057,
                head_sha="1ac4fc40",
                status="in_progress",
                created_at="2026-08-24T16:19:46Z",
            ),
            _run(
                32750203422,
                head_sha="814cc0fe",
                status="queued",
                created_at="2026-08-24T16:19:55Z",
                job_count=0,
            ),
        ],
        now=datetime(2026, 8, 24, 16, 36, 0, tzinfo=UTC),  # ~16 min after newest
        stale_after_seconds=900,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.run_id == 32750189057
    assert candidate.reason == "stale_head_blocking_concurrency_group"
    assert candidate.blocked_run_id == 32750203422
    assert candidate.blocked_run_pending_seconds >= 900


def test_does_not_flag_when_newer_run_is_below_the_stale_threshold() -> None:
    """A newer run queued for only a few seconds is ordinary queueing
    latency, not a wedge -- must not be force-cancelled prematurely."""
    candidates = determine_zombie_cancellations(
        runs=[
            _run(1, status="in_progress", created_at="2026-08-24T16:19:46Z"),
            _run(2, status="queued", created_at="2026-08-24T16:19:55Z", job_count=0),
        ],
        now=datetime(2026, 8, 24, 16, 20, 5, tzinfo=UTC),  # 10s after newest
        stale_after_seconds=900,
    )

    assert candidates == []


def test_does_not_flag_when_newer_run_has_scheduled_jobs() -> None:
    """Once the newer run has jobs scheduled, the group released normally --
    nothing to cancel."""
    candidates = determine_zombie_cancellations(
        runs=[
            _run(1, status="in_progress", created_at="2026-08-24T16:19:46Z"),
            _run(2, status="queued", created_at="2026-08-24T16:19:55Z", job_count=5),
        ],
        now=_NOW,
        stale_after_seconds=900,
    )

    assert candidates == []


def test_does_not_flag_single_run_groups() -> None:
    """One active run for a branch is normal steady state -- nothing shares
    its concurrency group, so there is nothing to unblock."""
    candidates = determine_zombie_cancellations(
        runs=[_run(1, status="queued", created_at="2026-08-24T16:19:55Z", job_count=0)],
        now=_NOW,
        stale_after_seconds=900,
    )

    assert candidates == []


def test_ignores_completed_runs_in_the_group() -> None:
    candidates = determine_zombie_cancellations(
        runs=[
            _run(1, status="completed", created_at="2026-08-24T16:19:46Z"),
            _run(2, status="queued", created_at="2026-08-24T16:19:55Z", job_count=0),
        ],
        now=_NOW,
        stale_after_seconds=900,
    )

    assert candidates == []


def test_groups_are_scoped_per_repo_and_head_branch() -> None:
    """Two different branches (even same repo) never block each other, and
    the same branch name in two different repos is a distinct group."""
    candidates = determine_zombie_cancellations(
        runs=[
            _run(
                1, repo="omnimarket", head_branch="jonah/omn-100", status="in_progress"
            ),
            _run(
                2,
                repo="omnimarket",
                head_branch="jonah/omn-200",
                status="queued",
                created_at="2026-08-24T16:19:55Z",
                job_count=0,
            ),
            _run(
                3,
                repo="onex_change_control",
                head_branch="jonah/omn-100",
                status="in_progress",
            ),
        ],
        now=_NOW,
        stale_after_seconds=900,
    )

    assert candidates == []


def test_preserves_the_newest_run_id_as_blocked_run() -> None:
    """With 3+ active runs for one branch, every older run is a candidate and
    all point at the single newest (blocked) run."""
    candidates = determine_zombie_cancellations(
        runs=[
            _run(1, status="in_progress", created_at="2026-08-24T16:19:00Z"),
            _run(2, status="in_progress", created_at="2026-08-24T16:19:30Z"),
            _run(
                3,
                status="queued",
                created_at="2026-08-24T16:19:55Z",
                job_count=0,
            ),
        ],
        now=datetime(2026, 8, 24, 16, 36, 0, tzinfo=UTC),
        stale_after_seconds=900,
    )

    assert {candidate.run_id for candidate in candidates} == {1, 2}
    assert all(candidate.blocked_run_id == 3 for candidate in candidates)


def test_empty_pull_requests_field_does_not_matter_grouping_is_by_head_branch() -> None:
    """Regression guard for the exact defect the OMN-16494 incident replay
    caught: the GitHub Actions runs API's `pull_requests[]` field came back
    empty for BOTH runs of the real omnimarket#2106 incident even though
    neither was a fork PR. `PullRequestRun` never carries a pr_number field
    at all (removed, not just unused) -- grouping is head_branch-only, so
    this failure mode is structurally impossible, not merely worked around."""
    assert not hasattr(
        PullRequestRun("r", 1, "b", "sha", "queued", "2026-01-01T00:00:00Z"),
        "pr_number",
    )

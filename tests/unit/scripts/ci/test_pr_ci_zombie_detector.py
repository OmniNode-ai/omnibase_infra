# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.ci import pr_ci_zombie_detector as detector
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


# --- OMN-19258: the scan fails loud when it could not see ---------------------
#
# Before OMN-19258, main() returned 0 on every completed scan. 68 of 2,904
# retained scheduled reports (2026-08-25..2026-09-23) carried a rate-limit fetch
# error for BOTH target repos and the job was green each time.

_REPOS = ["onex_change_control", "omnimarket"]


def _page(*items: dict[str, object]) -> str:
    return "\n".join(json.dumps(item) for item in items) + "\n"


def _quiet_page() -> str:
    """A full page with CI history and no active pull_request run."""
    return _page(
        {
            "id": 1,
            "event": "push",
            "status": "completed",
            "created_at": "2026-08-24T16:00:00Z",
            "head_sha": "a",
            "head_branch": "dev",
        },
        {
            "id": 2,
            "event": "pull_request",
            "status": "completed",
            "created_at": "2026-08-24T16:01:00Z",
            "head_sha": "b",
            "head_branch": "feature",
        },
    )


def _wedged_page() -> str:
    """The omnimarket#2106 shape: an old-head run holds the group, the new head sits."""
    return _page(
        {
            "id": 100,
            "event": "pull_request",
            "status": "in_progress",
            "created_at": "2020-01-01T00:00:00Z",
            "head_sha": "old",
            "head_branch": _BRANCH,
        },
        {
            "id": 101,
            "event": "pull_request",
            "status": "queued",
            "created_at": "2020-01-01T00:05:00Z",
            "head_sha": "new",
            "head_branch": _BRANCH,
        },
    )


def _fake_gh(pages: dict[str, str | Exception]) -> Callable[[list[str]], str]:
    def run_gh(args: list[str]) -> str:
        path = args[1]
        if path.endswith("/jobs"):
            return "0\n"
        repo = path.split("/")[3]
        page = pages[repo]
        if isinstance(page, Exception):
            raise page
        return page

    return run_gh


def _rate_limited() -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(
        1, ["gh"], output="", stderr="gh: API rate limit exceeded (HTTP 403)"
    )


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pages: dict[str, str | Exception],
    *extra: str,
) -> tuple[int, dict[str, object]]:
    monkeypatch.setattr(detector, "_run_gh", _fake_gh(pages))
    report = tmp_path / "report.json"
    argv = [arg for repo in _REPOS for arg in ("--repo", repo)]
    code = detector.main([*argv, "--report", str(report), *extra])
    assert report.is_file(), "the report must be written even when the scan fails"
    return code, json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_clean_scan_exits_zero_and_records_scanned_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, report = _run_main(
        monkeypatch,
        tmp_path,
        {"onex_change_control": _quiet_page(), "omnimarket": _quiet_page()},
    )
    assert code == 0
    assert report["ok"] is True
    assert report["fetch_errors"] == {}
    assert report["scanned"] == {
        "onex_change_control": {"runs_read": 2, "active_pull_request_runs": 0},
        "omnimarket": {"runs_read": 2, "active_pull_request_runs": 0},
    }


@pytest.mark.unit
def test_fetch_error_on_any_repo_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, report = _run_main(
        monkeypatch,
        tmp_path,
        {"onex_change_control": _rate_limited(), "omnimarket": _quiet_page()},
    )
    assert code == 1, "a scan that could not read a target repo must not exit 0"
    assert report["ok"] is False
    fetch_errors = report["fetch_errors"]
    assert isinstance(fetch_errors, dict)
    assert "rate limit" in fetch_errors["onex_change_control"]
    scanned = report["scanned"]
    assert isinstance(scanned, dict)
    assert "omnimarket" in scanned, "the readable repo is still scanned"


@pytest.mark.unit
def test_fetch_error_on_both_repos_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The measured 68-run shape: both repos rate-limited, nothing seen."""
    code, report = _run_main(
        monkeypatch,
        tmp_path,
        {"onex_change_control": _rate_limited(), "omnimarket": _rate_limited()},
        "--apply",
    )
    assert code == 1
    assert report["scanned"] == {}
    assert report["cancellations"] == []


@pytest.mark.unit
def test_zero_runs_read_is_a_blind_read_not_a_quiet_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, report = _run_main(
        monkeypatch,
        tmp_path,
        {"onex_change_control": "", "omnimarket": _quiet_page()},
    )
    assert code == 1
    scanned = report["scanned"]
    assert isinstance(scanned, dict)
    assert scanned["onex_change_control"] == {
        "runs_read": 0,
        "active_pull_request_runs": 0,
    }
    fetch_errors = report["fetch_errors"]
    assert isinstance(fetch_errors, dict)
    assert "onex_change_control" in fetch_errors


@pytest.mark.unit
def test_wedge_is_found_through_main_and_cancelled_on_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control for the scan path: the event/status filter now runs in
    Python, so a wedge must still reach the decision function."""
    cancelled: list[tuple[str, int]] = []
    monkeypatch.setattr(
        detector,
        "force_cancel_run",
        lambda owner, repo, run_id: cancelled.append((repo, run_id)),
    )
    code, report = _run_main(
        monkeypatch,
        tmp_path,
        {"onex_change_control": _quiet_page(), "omnimarket": _wedged_page()},
        "--apply",
    )
    assert code == 0
    assert cancelled == [("omnimarket", 100)]
    scanned = report["scanned"]
    assert isinstance(scanned, dict)
    assert scanned["omnimarket"] == {"runs_read": 2, "active_pull_request_runs": 2}


@pytest.mark.unit
def test_failed_force_cancel_exits_nonzero_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(owner: str, repo: str, run_id: int) -> None:
        raise subprocess.CalledProcessError(
            1, ["gh"], stderr=b"Resource not accessible by integration (HTTP 403)"
        )

    monkeypatch.setattr(detector, "force_cancel_run", refuse)
    code, report = _run_main(
        monkeypatch,
        tmp_path,
        {"onex_change_control": _quiet_page(), "omnimarket": _wedged_page()},
        "--apply",
    )
    assert code == 1
    cancel_errors = report["cancel_errors"]
    assert isinstance(cancel_errors, dict)
    assert "HTTP 403" in cancel_errors["omnimarket#100"]

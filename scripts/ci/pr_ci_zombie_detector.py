#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Force-cancel stale-head PR runs that wedge a shared CI concurrency group (OMN-16494).

Why this exists
----------------
`ready_for_review` (undraft) and a branch refresh (push/update-branch/synchronize)
both trigger CI for the same PR. Two independent mechanisms can leave the
*current*-head run stuck `status=queued` with zero scheduled jobs, forever,
while an *older*-head run for the same PR keeps holding the shared
concurrency group:

1. ``cancel-in-progress: true`` cancels most of the superseded run, but 1-2
   jobs on self-hosted runners can survive the cancel signal (observed:
   ``Pre-commit``, ``Contract Compliance Check`` in onex_change_control#7009
   and #7018) and keep the group held.
2. ``cancel-in-progress: false`` (single-flight queueing, e.g. omnimarket's
   OMN-14645 policy) never attempts to cancel anything -- the newer run
   legitimately waits for the older run's slowest job to finish, which can
   take a long time when that job runs long or stalls (verified live on
   omnimarket#2106: the new-head run's earliest job did not start until
   16:26:29Z -- 6m34s after the run was created at 16:19:55Z, and 4s after
   the old head's straggler jobs, ``Contract Compliance Check`` and
   ``Coverage Sweep Gate``, finished at 16:26:25Z; see
   ``tests/incident_replays/registry.yaml``'s ``omn16494-*`` cases for the
   captured proof).

Both mechanisms produce the identical externally-observable symptom this
detector keys on: the run for the CURRENT (newest) head of a PR is
``status=queued`` with zero jobs scheduled, for longer than
``--stale-after-seconds``, while an OLDER, still-non-terminal run exists for
the same PR. That older run is the "stale-head" run; force-cancelling it
releases the concurrency group and lets the current-head run proceed. This
mirrors the manual ``POST .../actions/runs/{run_id}/force-cancel`` remedy
used 3x by hand before this detector existed (OMN-16494).

Grouped by head branch, not PR number (a correctness fix the incident replay
itself caught): the GitHub Actions runs API's ``pull_requests[]`` field --
the obvious way to key a run to "which PR is this" -- comes back an EMPTY
array for both runs of the real omnimarket#2106 incident this detector
targets, even though neither is a fork PR. A same-repo PR is not a
documented case for that field going empty, but it happens (grep any recent
`pull_request`-event run in omnimarket or onex_change_control and the
`pull_requests` array is reliably ``[]``). ``head_branch`` has no such gap --
it is always populated for a `pull_request`-event run -- and it is a safe
grouping key on its own: GitHub does not allow two open PRs in the same repo
to share a head branch, so (repo, head_branch) identifies exactly the runs
this detector needs grouped.

Distinct from ``merge_queue_janitor.py``: that script cancels superseded
``merge_group`` runs; this one cancels stale-head ``pull_request`` runs. The
decision function here is intentionally symmetric to it (same
dataclass-in/dataclass-out shape, same dry-run-by-default posture) so both
janitors read the same way.

Usage
-----
    pr_ci_zombie_detector.py --owner OmniNode-ai \\
        --repo onex_change_control --repo omnimarket \\
        --stale-after-seconds 900 --apply --report report.json

Exit code is always 0 on a completed scan (dry-run or apply); a per-repo
fetch failure is recorded in the report and printed as a warning, and does
not abort the scan of the remaining repos.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ACTIVE_RUN_STATUSES = frozenset({"queued", "in_progress"})
DEFAULT_STALE_AFTER_SECONDS = 900.0  # 15 minutes


@dataclass(frozen=True)
class PullRequestRun:
    repo: str
    run_id: int
    head_branch: str
    head_sha: str
    status: str
    created_at: str
    job_count: int | None = None
    """Total scheduled jobs. ``None`` means "not fetched" (only fetched for
    the newest run per (repo, head_branch) group, since that is the only run
    whose job count the decision depends on)."""


@dataclass(frozen=True)
class CancellationCandidate:
    repo: str
    run_id: int
    head_branch: str
    head_sha: str
    reason: str
    blocked_run_id: int
    """The newer-head run this cancellation unblocks."""
    blocked_run_pending_seconds: float


def determine_zombie_cancellations(
    *,
    runs: list[PullRequestRun],
    now: datetime,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> list[CancellationCandidate]:
    """Return stale-head runs safe to force-cancel.

    For each (repo, head_branch) group with 2+ active runs: if the newest run
    is ``queued`` with zero scheduled jobs and has been in that state for at
    least ``stale_after_seconds``, every OLDER run in the group that is still
    non-terminal (``queued`` or ``in_progress``) is a stale-head candidate --
    it is what is holding the shared concurrency group open.

    A newest run with a non-zero job count, or one that is not ``queued``, is
    proceeding normally (the group released on its own) and yields no
    candidates. A newest run that is merely young (below the threshold) is
    treated as ordinary queueing latency, not a wedge.
    """
    grouped: dict[tuple[str, str], list[PullRequestRun]] = {}
    for run in runs:
        if run.status not in ACTIVE_RUN_STATUSES:
            continue
        grouped.setdefault((run.repo, run.head_branch), []).append(run)

    candidates: list[CancellationCandidate] = []
    for (_repo, _head_branch), group_runs in grouped.items():
        if len(group_runs) < 2:
            continue
        newest = max(group_runs, key=_run_sort_key)
        if newest.status != "queued":
            continue
        if newest.job_count is None or newest.job_count != 0:
            continue
        pending_seconds = (now - _parse_timestamp(newest.created_at)).total_seconds()
        if pending_seconds < stale_after_seconds:
            continue
        for run in group_runs:
            if run.run_id == newest.run_id:
                continue
            candidates.append(
                CancellationCandidate(
                    repo=run.repo,
                    run_id=run.run_id,
                    head_branch=run.head_branch,
                    head_sha=run.head_sha,
                    reason="stale_head_blocking_concurrency_group",
                    blocked_run_id=newest.run_id,
                    blocked_run_pending_seconds=pending_seconds,
                )
            )

    return sorted(candidates, key=lambda candidate: (candidate.repo, candidate.run_id))


def _run_sort_key(run: PullRequestRun) -> tuple[str, int]:
    return (run.created_at, run.run_id)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# I/O layer -- kept thin and untested at unit level; the decision function
# above carries the logic and is what the unit tests exercise.
# ---------------------------------------------------------------------------


def _run_gh(args: list[str]) -> str:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def fetch_active_pull_request_runs(
    owner: str, repo: str, limit: int
) -> list[PullRequestRun]:
    """Fetch active (queued/in_progress) `pull_request`-event runs for `repo`.

    Deliberately a single, un-paginated page: the GitHub Actions runs API
    returns most-recent-first, and any run this detector cares about is by
    definition still active (queued/in_progress), so it is always within the
    first page for any repo whose CI throughput does not exceed ``limit``
    completed runs between scans. ``--paginate`` was tried and timed out --
    it walks the *entire* run history (years of completed runs) because the
    jq status filter matches almost nothing per page, not because active
    runs are deep in the history.

    ``gh api --jq`` streams one compact JSON object per matched item (not a
    single JSON array), so the output is parsed line-by-line -- the same
    convention ``handler_runner_fleet_snapshot.py`` uses for its queue probe.
    """
    stdout = _run_gh(
        [
            "api",
            f"/repos/{owner}/{repo}/actions/runs",
            "-X",
            "GET",
            "-f",
            f"per_page={limit}",
            "--jq",
            '.workflow_runs[] | select(.event=="pull_request" and '
            '(.status=="queued" or .status=="in_progress")) '
            "| {id,status,created_at,head_sha,head_branch}",
        ]
    )
    runs: list[PullRequestRun] = []
    for line in stdout.strip().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        head_branch = item.get("head_branch")
        if not head_branch:
            continue
        runs.append(
            PullRequestRun(
                repo=repo,
                run_id=int(item["id"]),
                head_branch=str(head_branch),
                head_sha=str(item.get("head_sha", "")),
                status=str(item["status"]),
                created_at=str(item["created_at"]),
            )
        )
    return runs


def fetch_job_count(owner: str, repo: str, run_id: int) -> int:
    stdout = _run_gh(
        [
            "api",
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
            "--jq",
            ".total_count",
        ]
    )
    return int(stdout.strip())


def with_newest_job_counts(
    owner: str, runs: list[PullRequestRun]
) -> list[PullRequestRun]:
    """Fetch job counts only for the newest run in each active (repo, branch) group.

    Job-count is a paid API call per run; the decision function only needs it
    for the newest run in each group, so this avoids fetching it for every
    active run.
    """
    grouped: dict[tuple[str, str], list[PullRequestRun]] = {}
    for run in runs:
        grouped.setdefault((run.repo, run.head_branch), []).append(run)

    newest_ids: set[tuple[str, int]] = set()
    for group_runs in grouped.values():
        if len(group_runs) < 2:
            continue
        newest = max(group_runs, key=_run_sort_key)
        newest_ids.add((newest.repo, newest.run_id))

    enriched: list[PullRequestRun] = []
    for run in runs:
        if (run.repo, run.run_id) not in newest_ids:
            enriched.append(run)
            continue
        job_count = fetch_job_count(owner, run.repo, run.run_id)
        enriched.append(
            PullRequestRun(
                repo=run.repo,
                run_id=run.run_id,
                head_branch=run.head_branch,
                head_sha=run.head_sha,
                status=run.status,
                created_at=run.created_at,
                job_count=job_count,
            )
        )
    return enriched


def force_cancel_run(owner: str, repo: str, run_id: int) -> None:
    subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "POST",
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/force-cancel",
        ],
        check=True,
        capture_output=True,
    )


def write_report(
    *,
    path: Path,
    owner: str,
    repos: list[str],
    apply: bool,
    stale_after_seconds: float,
    candidates: list[CancellationCandidate],
    fetch_errors: dict[str, str],
) -> None:
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "owner": owner,
        "repos": repos,
        "mode": "apply" if apply else "dry-run",
        "stale_after_seconds": stale_after_seconds,
        "cancellations": [asdict(candidate) for candidate in candidates],
        "fetch_errors": fetch_errors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="OmniNode-ai")
    parser.add_argument(
        "--repo",
        dest="repos",
        action="append",
        required=True,
        help="Target repo name (repeatable). e.g. --repo onex_change_control --repo omnimarket",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=DEFAULT_STALE_AFTER_SECONDS,
        help="How long the current-head run must sit queued with zero jobs "
        "before its older-head peer is force-cancelled.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Force-cancel candidates. Omit for dry-run mode.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON receipt path.")
    args = parser.parse_args(argv)

    all_runs: list[PullRequestRun] = []
    fetch_errors: dict[str, str] = {}
    for repo in args.repos:
        try:
            all_runs.extend(
                fetch_active_pull_request_runs(args.owner, repo, args.limit)
            )
        except subprocess.CalledProcessError as exc:
            fetch_errors[repo] = exc.stderr.strip()[:500] if exc.stderr else str(exc)
            print(
                f"[pr-ci-zombie-detector] WARNING: fetch failed for {repo}: {fetch_errors[repo]}"
            )

    enriched_runs: list[PullRequestRun] = []
    for repo in {run.repo for run in all_runs}:
        repo_runs = [run for run in all_runs if run.repo == repo]
        try:
            enriched_runs.extend(with_newest_job_counts(args.owner, repo_runs))
        except subprocess.CalledProcessError as exc:
            fetch_errors.setdefault(
                repo, exc.stderr.strip()[:500] if exc.stderr else str(exc)
            )
            enriched_runs.extend(repo_runs)

    candidates = determine_zombie_cancellations(
        runs=enriched_runs,
        now=datetime.now(UTC),
        stale_after_seconds=args.stale_after_seconds,
    )

    if not candidates:
        print("[pr-ci-zombie-detector] no wedged concurrency groups found")
    for candidate in candidates:
        print(
            "[pr-ci-zombie-detector] "
            f"{'force-cancel' if args.apply else 'would-force-cancel'} "
            f"repo={candidate.repo} run={candidate.run_id} branch={candidate.head_branch!r} "
            f"reason={candidate.reason} unblocks_run={candidate.blocked_run_id} "
            f"(pending {candidate.blocked_run_pending_seconds:.0f}s)"
        )
        if args.apply:
            force_cancel_run(args.owner, candidate.repo, candidate.run_id)

    if args.report:
        write_report(
            path=args.report,
            owner=args.owner,
            repos=args.repos,
            apply=args.apply,
            stale_after_seconds=args.stale_after_seconds,
            candidates=candidates,
            fetch_errors=fetch_errors,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

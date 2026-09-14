# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Decide whether a stale ``occ-preflight`` verdict must be re-run (OMN-18352).

Why this module exists
----------------------
``occ-preflight-heal.yml`` (OMN-16171 / OMN-14241) clears the failure that a PR
picks up when its ``Evidence-Source`` line arrives by a body edit *after* the
``occ-preflight`` job has already recorded a verdict. It heals in place, with
``gh run rerun --failed``, so the run id ``CI Summary`` polls is preserved and
two jobs move instead of the whole ~48-job matrix.

The guard used to live in the workflow as inline shell, and it required the
target CI run's own ``status`` to already read ``completed``. That precondition
is false at the only moment the heal was ever invoked: the OCC autobind stamp
lands about a minute after the initiating push, while the matrix is still
``in_progress``. Live on ``omnibase_infra#3503`` — heal job green at 20:53:08Z
having done nothing, ``occ-preflight / eligibility`` and ``CI Summary`` frozen
on the pre-stamp FAILURE, cleared 28 minutes later by a human
``gh run rerun --failed`` (run ``34782083407``, ``run_attempt: 2``). Reproduced
the same session on ``#3505``.

The fix is an invocation point, not a narrower window
----------------------------------------------------
"Not yet completed" is a *not yet*. The heal is now also invoked by the CI
run's own ``workflow_run: completed`` event — the moment its precondition
becomes true — while keeping ``pull_request: edited`` for a stamp that lands
after the run has already finished. Between them the two cover every ordering
of (stamp lands) and (run completes).

A bounded in-job poll was considered and rejected: the matrix runs for tens of
minutes, so a poll would pin a runner for the duration of someone else's CI and
would *still* lose the race whenever the matrix outran the ceiling. Waiting for
an event GitHub already emits costs nothing and cannot time out.

Recursion and waste bounds
--------------------------
Being re-invoked on run completion makes the guard recursion-exposed for the
first time: a heal-issued rerun completes, and that completion is itself a
``workflow_run`` event. Two independent conditions bound it.

*Staleness.* Re-run only when the failed preflight job STARTED BEFORE the PR's
``updated_at`` — i.e. the recorded verdict predates the PR state it claims to
describe. After a heal, the fresh preflight job starts later than
``updated_at``, so the next completion reads "not stale" and stops. The same
condition keeps the completion trigger from re-queueing a preflight that failed
for a reason no body edit touched, which the ``edited``-only trigger got for
free.

*Attempt ceiling.* A hard cap on ``run_attempt``, as a backstop that terminates
an unforeseen loop even if the timestamp reasoning above is ever wrong.

``started_at`` is deliberately the comparison point rather than
``completed_at``: the preflight reads the PR body live at some instant during
the job, so a body edit landing mid-job may or may not have been seen.
Comparing against the job's start treats that window as stale, which errs
toward healing — the cheap direction (two jobs) rather than the expensive one
(a wedged required context and a human rerun).

Failing loud beats failing silent
---------------------------------
The defect that produced this module was a green heal job that had healed
nothing: one "nothing to do" message covered both "there was nothing to heal"
and "I could not tell". Every refusal here carries its own
:class:`EnumHealOutcome`, and a PR whose state cannot be read at all exits
non-zero so the heal job goes red instead of reporting a clean no-op.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # fixed argv, no shell, trusted gh binary
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Protocol

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1

#: Backstop against an unforeseen re-trigger loop. Each heal moves two jobs, so
#: the worst case this admits is cheap; the staleness condition is the control
#: that is expected to do the actual work.
MAX_HEAL_RUN_ATTEMPT: Final[int] = 5

#: Must match ``ci.yml``'s ``name:``. Pinned by
#: ``tests/ci/test_occ_eval_path_trigger_coverage.py``.
DEFAULT_CI_WORKFLOW_NAME: Final[str] = "CI"

#: Must match ``ci.yml``'s ``occ-preflight`` job id. The reusable workflow
#: publishes its job as ``occ-preflight / eligibility``, so this is a prefix.
DEFAULT_PREFLIGHT_JOB_PREFIX: Final[str] = "occ-preflight"

_PAGE_SIZE: Final[int] = 100
_MAX_PAGES: Final[int] = 20


class EnumHealOutcome(StrEnum):
    """Why the heal did or did not issue a rerun. One value per branch."""

    RERUN_REQUIRED = "rerun_required"
    NO_CI_RUN = "no_ci_run"
    RUN_NOT_COMPLETED = "run_not_completed"
    NO_FAILED_PREFLIGHT = "no_failed_preflight"
    VERDICT_NOT_STALE = "verdict_not_stale"
    ATTEMPT_CEILING = "attempt_ceiling"
    NO_PULL_REQUEST = "no_pull_request"
    PR_STATE_UNRESOLVED = "pr_state_unresolved"


@dataclass(frozen=True)
class CiRunSnapshot:
    """The three fields of an Actions run this guard reasons about."""

    run_id: int
    status: str
    run_attempt: int
    head_sha: str


@dataclass(frozen=True)
class HealDecision:
    outcome: EnumHealOutcome
    run_id: int | None
    detail: str

    @property
    def rerun(self) -> bool:
        return self.outcome is EnumHealOutcome.RERUN_REQUIRED


class GhPort(Protocol):
    """The GitHub reads and the one write this guard needs."""

    def latest_ci_run(
        self, *, repo: str, head_sha: str, workflow_name: str
    ) -> CiRunSnapshot | None: ...

    def run_snapshot(self, *, repo: str, run_id: int) -> CiRunSnapshot | None: ...

    def earliest_failed_preflight_started_at(
        self, *, repo: str, run_id: int, job_prefix: str
    ) -> datetime | None: ...

    def pr_updated_at(self, *, repo: str, pr_number: int) -> datetime | None: ...

    def resolve_pr_number(self, *, repo: str, head_sha: str) -> int | None: ...

    def rerun_failed(self, *, repo: str, run_id: int) -> None: ...


def parse_github_timestamp(raw: str | None) -> datetime | None:
    """Parse an Actions/REST ISO-8601 timestamp into an aware UTC datetime."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def snapshot_from_run_payload(payload: object) -> CiRunSnapshot | None:
    """Read the three fields this guard needs out of an Actions run payload.

    Pure and public so an incident replay can drive it over the committed bytes
    of a real run rather than over a hand-shaped dict (OMN-15547 R1/R5).
    """
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("id")
    status = payload.get("status")
    attempt = payload.get("run_attempt", 1)
    head_sha = payload.get("head_sha", "")
    if not isinstance(run_id, int) or not isinstance(status, str):
        return None
    return CiRunSnapshot(
        run_id=run_id,
        status=status,
        run_attempt=attempt if isinstance(attempt, int) else 1,
        head_sha=head_sha if isinstance(head_sha, str) else "",
    )


def earliest_failed_preflight_started_at_in_jobs(
    payload: object, *, job_prefix: str
) -> datetime | None:
    """Earliest start among the FAILED preflight jobs in one jobs payload.

    Earliest rather than latest because it is the conservative choice: it makes
    a verdict look stale in every window where the body might have changed
    under the job. Healing spuriously costs two jobs; failing to heal costs a
    wedged required context and a human rerun.

    Pure and public for the same reason as :func:`snapshot_from_run_payload`.
    """
    if not isinstance(payload, dict):
        return None
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return None
    starts: list[datetime] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if job.get("conclusion") != "failure":
            continue
        name = job.get("name")
        if not isinstance(name, str) or not name.startswith(job_prefix):
            continue
        raw = job.get("started_at")
        started = parse_github_timestamp(raw if isinstance(raw, str) else None)
        if started is not None:
            starts.append(started)
    return min(starts) if starts else None


def decide_heal(
    *,
    run: CiRunSnapshot | None,
    earliest_failed_preflight_started_at: datetime | None,
    pr_updated_at: datetime | None,
    max_run_attempt: int = MAX_HEAL_RUN_ATTEMPT,
) -> HealDecision:
    """Pure verdict. Every refusal names itself.

    Order matters only for which reason gets reported when several apply; the
    most informative one is reported first.
    """
    if run is None:
        return HealDecision(
            outcome=EnumHealOutcome.NO_CI_RUN,
            run_id=None,
            detail="no CI run found for this head; nothing to heal",
        )

    if run.status != "completed":
        return HealDecision(
            outcome=EnumHealOutcome.RUN_NOT_COMPLETED,
            run_id=run.run_id,
            detail=(
                f"run {run.run_id} is {run.status}; a single failed job cannot be "
                f"re-run inside an unfinished run. The run's own completion event "
                f"invokes this guard again -- that is the OMN-18352 fix, and this "
                f"branch is no longer terminal."
            ),
        )

    if earliest_failed_preflight_started_at is None:
        return HealDecision(
            outcome=EnumHealOutcome.NO_FAILED_PREFLIGHT,
            run_id=run.run_id,
            detail=(
                f"run {run.run_id} has no failed preflight job; re-running every "
                f"failed job here would re-queue genuine test failures"
            ),
        )

    if pr_updated_at is None:
        return HealDecision(
            outcome=EnumHealOutcome.PR_STATE_UNRESOLVED,
            run_id=run.run_id,
            detail=(
                "could not read the pull request's updated_at, so staleness is "
                "undecidable; failing loud rather than reporting a clean no-op"
            ),
        )

    if earliest_failed_preflight_started_at >= pr_updated_at:
        return HealDecision(
            outcome=EnumHealOutcome.VERDICT_NOT_STALE,
            run_id=run.run_id,
            detail=(
                f"preflight started {earliest_failed_preflight_started_at.isoformat()} "
                f"at or after the PR's last update {pr_updated_at.isoformat()}: the "
                f"verdict describes the current PR state, so it is a real failure"
            ),
        )

    if run.run_attempt > max_run_attempt:
        return HealDecision(
            outcome=EnumHealOutcome.ATTEMPT_CEILING,
            run_id=run.run_id,
            detail=(
                f"run {run.run_id} is on attempt {run.run_attempt}, above the "
                f"ceiling of {max_run_attempt}; refusing to heal further so an "
                f"unforeseen re-trigger loop terminates"
            ),
        )

    return HealDecision(
        outcome=EnumHealOutcome.RERUN_REQUIRED,
        run_id=run.run_id,
        detail=(
            f"preflight started {earliest_failed_preflight_started_at.isoformat()} "
            f"before the PR's last update {pr_updated_at.isoformat()}: the recorded "
            f"verdict predates the body it describes"
        ),
    )


class GhCli:
    """:class:`GhPort` backed by the ``gh`` binary."""

    def _api(self, path: str) -> object:
        proc = subprocess.run(
            ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(
                f"gh api {path} failed (exit {proc.returncode}): {proc.stderr.strip()}",
                file=sys.stderr,
            )
            return None
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            print(f"gh api {path} returned unparseable JSON: {exc}", file=sys.stderr)
            return None

    @staticmethod
    def _snapshot(payload: object) -> CiRunSnapshot | None:
        return snapshot_from_run_payload(payload)

    def latest_ci_run(
        self, *, repo: str, head_sha: str, workflow_name: str
    ) -> CiRunSnapshot | None:
        payload = self._api(
            f"repos/{repo}/actions/runs?head_sha={head_sha}&per_page={_PAGE_SIZE}"
        )
        if not isinstance(payload, dict):
            return None
        runs = payload.get("workflow_runs")
        if not isinstance(runs, list):
            return None
        matching = [
            run
            for run in runs
            if isinstance(run, dict) and run.get("name") == workflow_name
        ]
        if not matching:
            return None
        matching.sort(key=lambda run: str(run.get("created_at", "")))
        return self._snapshot(matching[-1])

    def run_snapshot(self, *, repo: str, run_id: int) -> CiRunSnapshot | None:
        return self._snapshot(self._api(f"repos/{repo}/actions/runs/{run_id}"))

    def earliest_failed_preflight_started_at(
        self, *, repo: str, run_id: int, job_prefix: str
    ) -> datetime | None:
        """Paginate the latest attempt's job list and reduce over the pages."""
        starts: list[datetime] = []
        for page in range(1, _MAX_PAGES + 1):
            payload = self._api(
                f"repos/{repo}/actions/runs/{run_id}/jobs"
                f"?per_page={_PAGE_SIZE}&page={page}&filter=latest"
            )
            if not isinstance(payload, dict):
                return None
            jobs = payload.get("jobs")
            if not isinstance(jobs, list):
                return None
            page_earliest = earliest_failed_preflight_started_at_in_jobs(
                payload, job_prefix=job_prefix
            )
            if page_earliest is not None:
                starts.append(page_earliest)
            if len(jobs) < _PAGE_SIZE:
                break
        return min(starts) if starts else None

    def pr_updated_at(self, *, repo: str, pr_number: int) -> datetime | None:
        payload = self._api(f"repos/{repo}/pulls/{pr_number}")
        if not isinstance(payload, dict):
            return None
        raw = payload.get("updated_at")
        return parse_github_timestamp(raw if isinstance(raw, str) else None)

    def resolve_pr_number(self, *, repo: str, head_sha: str) -> int | None:
        payload = self._api(
            f"repos/{repo}/commits/{head_sha}/pulls?per_page={_PAGE_SIZE}"
        )
        if not isinstance(payload, list):
            return None
        candidates = [pr for pr in payload if isinstance(pr, dict)]
        for pr in candidates:
            if pr.get("state") == "open" and isinstance(pr.get("number"), int):
                number = pr["number"]
                assert isinstance(number, int)
                return number
        for pr in candidates:
            if isinstance(pr.get("number"), int):
                number = pr["number"]
                assert isinstance(number, int)
                return number
        return None

    def rerun_failed(self, *, repo: str, run_id: int) -> None:
        proc = subprocess.run(
            ["gh", "run", "rerun", str(run_id), "--failed", "--repo", repo],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"gh run rerun {run_id} --failed failed (exit {proc.returncode}): "
                f"{proc.stderr.strip()}"
            )
        print(proc.stdout.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name of the product repo")
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="CI run id (the workflow_run completion path knows it directly)",
    )
    parser.add_argument(
        "--head-sha",
        default=None,
        help="head SHA (the pull_request edited path resolves the run from it)",
    )
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--workflow-name", default=DEFAULT_CI_WORKFLOW_NAME)
    parser.add_argument("--preflight-job-prefix", default=DEFAULT_PREFLIGHT_JOB_PREFIX)
    parser.add_argument("--max-run-attempt", type=int, default=MAX_HEAL_RUN_ATTEMPT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="decide and report without issuing the rerun",
    )
    return parser


def main(argv: list[str] | None = None, *, gh: GhPort | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.run_id is None and not args.head_sha:
        print("one of --run-id or --head-sha is required", file=sys.stderr)
        return EXIT_ERROR

    client: GhPort = gh if gh is not None else GhCli()

    if args.run_id is not None:
        run = client.run_snapshot(repo=args.repo, run_id=args.run_id)
    else:
        run = client.latest_ci_run(
            repo=args.repo,
            head_sha=args.head_sha,
            workflow_name=args.workflow_name,
        )

    # The workflow_dispatch and workflow_run paths may know only the run id, so
    # fall back to the run's OWN head SHA rather than reporting "no pull
    # request" on exactly the events this guard exists to serve.
    pr_number: int | None = args.pr_number
    head_sha: str = args.head_sha or (run.head_sha if run is not None else "")
    if pr_number is None and head_sha:
        pr_number = client.resolve_pr_number(repo=args.repo, head_sha=head_sha)

    started_at: datetime | None = None
    if run is not None and run.status == "completed":
        started_at = client.earliest_failed_preflight_started_at(
            repo=args.repo,
            run_id=run.run_id,
            job_prefix=args.preflight_job_prefix,
        )

    if run is not None and run.status == "completed" and started_at is not None:
        if pr_number is None:
            decision = HealDecision(
                outcome=EnumHealOutcome.NO_PULL_REQUEST,
                run_id=run.run_id,
                detail=(
                    "no pull request is associated with this run (a push or "
                    "merge_group run); the heal only applies to PR runs"
                ),
            )
            print(f"[{decision.outcome.value}] {decision.detail}")
            return EXIT_OK
        updated_at = client.pr_updated_at(repo=args.repo, pr_number=pr_number)
    else:
        updated_at = None

    decision = decide_heal(
        run=run,
        earliest_failed_preflight_started_at=started_at,
        pr_updated_at=updated_at,
        max_run_attempt=args.max_run_attempt,
    )
    print(f"[{decision.outcome.value}] {decision.detail}")

    if decision.outcome is EnumHealOutcome.PR_STATE_UNRESOLVED:
        return EXIT_ERROR

    if not decision.rerun:
        return EXIT_OK

    if args.dry_run:
        print(f"--dry-run: would re-run failed jobs of run {decision.run_id}")
        return EXIT_OK

    assert decision.run_id is not None
    try:
        client.rerun_failed(repo=args.repo, run_id=decision.run_id)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    print(f"re-ran failed jobs of run {decision.run_id} in place")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mechanical dispatch throttle for bulk PR operations (OMN-16284).

Why this exists
----------------
At ~01:45Z on 2026-08-20 a merge-sweep lane armed/update-branched ~108 PRs
in one unthrottled burst. Update-branching triggers a full fresh check-suite
per PR; onex_change_control PRs alone carry ~63 checks each (~19 needing
self-hosted ``omnibase-ci`` runners). The queued-run count grew to ~1065,
the shared org-level runner pool (88 runners, no per-repo fair-share) sat
77-88/88 busy for ~4 hours, and every landing chain org-wide starved. The
burst settled ~95-100% RED from incident-window transients: 1 merge out of
~108. The rule ("throttle, serialize heavy work") existed only in prose and
did not bind — same failure class as ``feedback_a_rule_is_not_a_mechanism``.

This script is the mechanism. ALL bulk PR operations (update-branch,
arm-automerge, mass reruns, mass body edits) should route through it rather
than a hand-rolled loop over ``gh``.

Mechanical guard (no bypass)
-----------------------------
The queue-depth gate (:func:`wait_for_queue_depth`) has no force/skip/bypass
parameter anywhere in this module or its CLI. A caller cannot opt out of
throttling short of not using this tool at all — see
``docs/runbooks/bulk-pr-operations.md`` for the doctrine-wiring follow-up
that makes *not using it* visible.

Usage
-----
    bulk_pr_throttle.py --owner OmniNode-ai --repo onex_change_control \\
        --prs 6751,6752,6753 --operation rerun-failed --dry-run

    bulk_pr_throttle.py --owner OmniNode-ai --repo onex_change_control \\
        --prs 6751,6752,6753 --operation rerun-failed \\
        --wave-size 5 --queue-depth-threshold 150

Exit codes: 0 = all waves completed with all PR operations succeeding
(or a dry-run plan was printed), 1 = refused (bad input / cap exceeded /
queue-depth timeout) or at least one PR operation failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Tunables. Every numeric default here is deliberately conservative — the
# 2026-08-20 incident was caused by the ABSENCE of these defaults, not by
# them being set wrong.
# ---------------------------------------------------------------------------

DEFAULT_WAVE_SIZE = 10
#: Hard ceiling on --wave-size. Not flag-overridable past this point — the
#: mechanism must not be defeatable by simply passing a huge wave size.
MAX_WAVE_SIZE = 25
DEFAULT_QUEUE_DEPTH_THRESHOLD = 150
#: Default cap on total PRs processed in one invocation. Exceeding this
#: without explicitly passing --max-total-prs is refused (rule: "no silent
#: defaults", applied to batch size rather than owner/repo).
DEFAULT_MAX_TOTAL_PRS = 50
DEFAULT_BLOCK_POLL_SECONDS = 30.0
#: Maximum total time this tool will block waiting for queue depth to drop
#: before a single wave. A gate that can block forever is indistinguishable
#: from a hang; this raises QueueDepthTimeoutError instead of proceeding.
DEFAULT_MAX_BLOCK_SECONDS = 1800.0
DEFAULT_GH_TIMEOUT_SECONDS = 120.0

VALID_OPERATIONS = (
    "update-branch",
    "arm-automerge",
    "rerun-failed",
    "noop-dry-run",
)


class BulkPrThrottleError(RuntimeError):
    """Base class for refusal / fatal errors raised by this tool."""


class TotalPrLimitExceededError(BulkPrThrottleError):
    """Raised when the PR batch exceeds the cap without an explicit override."""


class QueueDepthTimeoutError(BulkPrThrottleError):
    """Raised when queue depth stays above threshold past max_wait_seconds."""


@dataclass(frozen=True)
class PrOutcome:
    pr_number: int
    success: bool
    detail: str


@dataclass(frozen=True)
class WaveReceipt:
    wave_index: int
    pr_numbers: tuple[int, ...]
    operation: str
    dry_run: bool
    queue_depth_before: int
    queue_depth_after: int | None
    started_at: str
    completed_at: str
    outcomes: tuple[PrOutcome, ...]


@dataclass(frozen=True)
class BulkRunReport:
    owner: str
    repo: str
    operation: str
    wave_size: int
    queue_depth_threshold: int
    dry_run: bool
    waves: tuple[WaveReceipt, ...]


class PartialBulkRunError(BulkPrThrottleError):
    """Raised when a bulk run aborts after one or more receiptable waves."""

    def __init__(self, message: str, report: BulkRunReport) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Wave partitioning
# ---------------------------------------------------------------------------


def partition_into_waves(
    pr_numbers: Sequence[int], wave_size: int
) -> list[tuple[int, ...]]:
    """Split ``pr_numbers`` into consecutive waves of at most ``wave_size``."""
    if wave_size < 1:
        raise ValueError(f"wave_size must be >= 1, got {wave_size}")
    if wave_size > MAX_WAVE_SIZE:
        raise ValueError(
            f"wave_size {wave_size} exceeds the hard ceiling of {MAX_WAVE_SIZE} "
            "— the ceiling exists so a bulk operation can never dispatch an "
            "unthrottled burst regardless of flags"
        )
    items = list(pr_numbers)
    return [tuple(items[i : i + wave_size]) for i in range(0, len(items), wave_size)]


# ---------------------------------------------------------------------------
# Refusal: total PR cap
# ---------------------------------------------------------------------------


def validate_total_prs(
    pr_numbers: Sequence[int],
    *,
    max_total_prs: int,
    explicit_max_total_prs: bool,
) -> None:
    """Refuse to process more than ``max_total_prs`` PRs.

    ``explicit_max_total_prs`` changes the refusal wording only. It is not a
    bypass; a caller can raise the cap explicitly, but the raised cap remains a
    real ceiling.
    """
    if len(pr_numbers) <= max_total_prs:
        return
    if explicit_max_total_prs:
        raise TotalPrLimitExceededError(
            f"refusing to process {len(pr_numbers)} PRs against the explicit "
            f"cap of {max_total_prs}. Pass --max-total-prs {len(pr_numbers)} "
            "(or higher) explicitly if this larger batch is intentional."
        )
    raise TotalPrLimitExceededError(
        f"refusing to process {len(pr_numbers)} PRs against the default "
        f"cap of {max_total_prs}. Pass --max-total-prs {len(pr_numbers)} "
        "(or higher) explicitly to raise the cap — this is a fail-fast "
        "guard against an accidental unthrottled burst, not a hard limit."
    )


# ---------------------------------------------------------------------------
# Threshold blocking (the queue-depth gate — no bypass parameter exists)
# ---------------------------------------------------------------------------


def wait_for_queue_depth(
    *,
    get_queue_depth: Callable[[], int],
    threshold: int,
    poll_seconds: float,
    max_wait_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda msg: None,
) -> int:
    """Block, polling ``get_queue_depth``, until depth <= threshold.

    Raises :class:`QueueDepthTimeoutError` rather than proceeding if the
    queue never drops within ``max_wait_seconds`` — a persistently saturated
    fleet is exactly the condition this tool must refuse to dispatch into.
    """
    if poll_seconds <= 0:
        raise ValueError(f"poll_seconds must be > 0, got {poll_seconds}")
    waited = 0.0
    depth = get_queue_depth()
    while depth > threshold:
        if waited >= max_wait_seconds:
            raise QueueDepthTimeoutError(
                f"queue depth {depth} still above threshold {threshold} after "
                f"waiting {waited:.0f}s (max_wait_seconds={max_wait_seconds}) "
                "— aborting rather than dispatching into a saturated fleet"
            )
        log(
            f"queue depth {depth} > threshold {threshold}; "
            f"blocking {poll_seconds:.0f}s before re-poll"
        )
        sleep_fn(poll_seconds)
        waited += poll_seconds
        depth = get_queue_depth()
    return depth


# ---------------------------------------------------------------------------
# Core wave-gated run
# ---------------------------------------------------------------------------


def run_bulk_operation(
    *,
    owner: str,
    repo: str,
    pr_numbers: Sequence[int],
    operation: str,
    wave_size: int = DEFAULT_WAVE_SIZE,
    queue_depth_threshold: int = DEFAULT_QUEUE_DEPTH_THRESHOLD,
    max_total_prs: int = DEFAULT_MAX_TOTAL_PRS,
    explicit_max_total_prs: bool = False,
    dry_run: bool = False,
    get_queue_depth: Callable[[], int] | None = None,
    apply_pr_operation: Callable[[str, str, int, str], PrOutcome] | None = None,
    poll_seconds: float = DEFAULT_BLOCK_POLL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_BLOCK_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: Callable[[str], None] = print,
) -> BulkRunReport:
    """Process ``pr_numbers`` through ``operation`` in bounded, queue-gated waves."""
    if not owner:
        raise BulkPrThrottleError(
            "owner must be explicitly provided (no silent default)"
        )
    if not repo:
        raise BulkPrThrottleError(
            "repo must be explicitly provided (no silent default)"
        )
    if operation not in VALID_OPERATIONS:
        raise BulkPrThrottleError(
            f"unknown operation {operation!r}; must be one of {VALID_OPERATIONS}"
        )
    if not pr_numbers:
        raise BulkPrThrottleError("pr_numbers must be non-empty")

    validate_total_prs(
        pr_numbers,
        max_total_prs=max_total_prs,
        explicit_max_total_prs=explicit_max_total_prs,
    )
    waves = partition_into_waves(list(pr_numbers), wave_size)

    if dry_run:
        log(
            f"[bulk-pr-throttle] DRY-RUN plan: {len(pr_numbers)} PR(s) across "
            f"{len(waves)} wave(s) of <= {wave_size}, operation={operation}, "
            f"owner={owner}, repo={repo}, queue_depth_threshold={queue_depth_threshold}"
        )
        for idx, wave in enumerate(waves, start=1):
            log(f"[bulk-pr-throttle]   wave {idx}: {list(wave)}")
        ts = now_fn().isoformat()
        return BulkRunReport(
            owner=owner,
            repo=repo,
            operation=operation,
            wave_size=wave_size,
            queue_depth_threshold=queue_depth_threshold,
            dry_run=True,
            waves=tuple(
                WaveReceipt(
                    wave_index=idx,
                    pr_numbers=wave,
                    operation=operation,
                    dry_run=True,
                    queue_depth_before=-1,
                    queue_depth_after=-1,
                    started_at=ts,
                    completed_at=ts,
                    outcomes=(),
                )
                for idx, wave in enumerate(waves, start=1)
            ),
        )

    if get_queue_depth is None or apply_pr_operation is None:
        raise BulkPrThrottleError(
            "get_queue_depth and apply_pr_operation are required outside dry-run mode"
        )

    wave_receipts: list[WaveReceipt] = []

    def partial_report() -> BulkRunReport:
        return BulkRunReport(
            owner=owner,
            repo=repo,
            operation=operation,
            wave_size=wave_size,
            queue_depth_threshold=queue_depth_threshold,
            dry_run=False,
            waves=tuple(wave_receipts),
        )

    for idx, wave in enumerate(waves, start=1):
        started_at = now_fn().isoformat()
        try:
            depth_before = wait_for_queue_depth(
                get_queue_depth=get_queue_depth,
                threshold=queue_depth_threshold,
                poll_seconds=poll_seconds,
                max_wait_seconds=max_wait_seconds,
                sleep_fn=sleep_fn,
                log=log,
            )
        except BulkPrThrottleError as exc:
            if wave_receipts:
                raise PartialBulkRunError(str(exc), partial_report()) from exc
            raise
        log(
            f"[bulk-pr-throttle] wave {idx}/{len(waves)}: depth_before={depth_before} "
            f"count={len(wave)} prs={list(wave)} operation={operation}"
        )
        outcomes = tuple(apply_pr_operation(owner, repo, pr, operation) for pr in wave)
        for outcome in outcomes:
            log(
                f"[bulk-pr-throttle]   pr={outcome.pr_number} "
                f"success={outcome.success} detail={outcome.detail}"
            )
        completed_at = now_fn().isoformat()
        try:
            depth_after = get_queue_depth()
        except BulkPrThrottleError as exc:
            wave_receipts.append(
                WaveReceipt(
                    wave_index=idx,
                    pr_numbers=wave,
                    operation=operation,
                    dry_run=False,
                    queue_depth_before=depth_before,
                    queue_depth_after=None,
                    started_at=started_at,
                    completed_at=completed_at,
                    outcomes=outcomes,
                )
            )
            raise PartialBulkRunError(str(exc), partial_report()) from exc
        log(f"[bulk-pr-throttle] wave {idx}/{len(waves)}: depth_after={depth_after}")
        wave_receipts.append(
            WaveReceipt(
                wave_index=idx,
                pr_numbers=wave,
                operation=operation,
                dry_run=False,
                queue_depth_before=depth_before,
                queue_depth_after=depth_after,
                started_at=started_at,
                completed_at=completed_at,
                outcomes=outcomes,
            )
        )

    return BulkRunReport(
        owner=owner,
        repo=repo,
        operation=operation,
        wave_size=wave_size,
        queue_depth_threshold=queue_depth_threshold,
        dry_run=False,
        waves=tuple(wave_receipts),
    )


def write_receipt(report: BulkRunReport, path: Path) -> None:
    """Serialize ``report`` as JSON to ``path`` (parent dirs created as needed)."""
    payload = {
        "owner": report.owner,
        "repo": report.repo,
        "operation": report.operation,
        "wave_size": report.wave_size,
        "queue_depth_threshold": report.queue_depth_threshold,
        "dry_run": report.dry_run,
        "waves": [
            {
                "wave_index": wave.wave_index,
                "pr_numbers": list(wave.pr_numbers),
                "pr_count": len(wave.pr_numbers),
                "operation": wave.operation,
                "dry_run": wave.dry_run,
                "queue_depth_before": wave.queue_depth_before,
                "queue_depth_after": wave.queue_depth_after,
                "started_at": wave.started_at,
                "completed_at": wave.completed_at,
                "outcomes": [asdict(o) for o in wave.outcomes],
            }
            for wave in report.waves
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# gh CLI integration (production seam — tests monkeypatch _run_gh, never hit
# the real GitHub API)
# ---------------------------------------------------------------------------


def _run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # nosec B603 - fixed argv, no shell, trusted gh binary
            ["gh", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=DEFAULT_GH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=124,
            stdout=exc.stdout or "",
            stderr=f"gh command timed out after {DEFAULT_GH_TIMEOUT_SECONDS:.0f}s",
        )


def gh_queue_depth(owner: str, repo: str) -> int:
    """Live queued-run count for ``owner/repo`` via the gh CLI."""
    result = _run_gh(
        [
            "api",
            f"repos/{owner}/{repo}/actions/runs?status=queued",
            "--jq",
            ".total_count",
        ]
    )
    if result.returncode != 0:
        raise BulkPrThrottleError(
            f"gh api queued-run count failed: {result.stderr.strip()}"
        )
    return int(result.stdout.strip())


def gh_apply_pr_operation(
    owner: str, repo: str, pr_number: int, operation: str
) -> PrOutcome:
    """Dispatch one PR-scoped operation via the gh CLI."""
    if operation == "update-branch":
        result = _run_gh(
            [
                "api",
                "-X",
                "PUT",
                f"repos/{owner}/{repo}/pulls/{pr_number}/update-branch",
            ]
        )
        return PrOutcome(
            pr_number=pr_number,
            success=result.returncode == 0,
            detail=(result.stdout or result.stderr).strip(),
        )
    if operation == "arm-automerge":
        result = _run_gh(
            [
                "pr",
                "merge",
                str(pr_number),
                "--repo",
                f"{owner}/{repo}",
                "--squash",
                "--auto",
            ]
        )
        return PrOutcome(
            pr_number=pr_number,
            success=result.returncode == 0,
            detail=(result.stdout or result.stderr).strip(),
        )
    if operation == "rerun-failed":
        return _gh_rerun_failed(owner, repo, pr_number)
    if operation == "noop-dry-run":
        return PrOutcome(pr_number=pr_number, success=True, detail="noop")
    raise BulkPrThrottleError(f"unknown operation {operation!r}")


def _gh_rerun_failed(owner: str, repo: str, pr_number: int) -> PrOutcome:
    """Find the PR's head-SHA workflow runs and rerun failed terminal runs.

    GitHub Actions exposes runner-loss / timeout incident-window reds as both
    ``failure`` and ``cancelled`` workflow-run conclusions. The operation name
    intentionally stays ``rerun-failed`` because GitHub's endpoint is
    ``rerun-failed-jobs``; the selector includes terminal cancelled runs so the
    throttle can clear the exact cancellation-heavy incident class it was built
    to control.
    """
    head_result = _run_gh(
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            f"{owner}/{repo}",
            "--json",
            "headRefOid",
        ]
    )
    if head_result.returncode != 0:
        return PrOutcome(
            pr_number=pr_number,
            success=False,
            detail=f"pr view failed: {head_result.stderr.strip()}",
        )
    try:
        head_sha = json.loads(head_result.stdout)["headRefOid"]
    except (json.JSONDecodeError, KeyError) as exc:
        return PrOutcome(
            pr_number=pr_number,
            success=False,
            detail=f"could not parse head SHA: {exc}",
        )

    runs: list[dict[str, object]] = []
    page = 1
    total_count: int | None = None
    while total_count is None or len(runs) < total_count:
        runs_result = _run_gh(
            [
                "api",
                f"repos/{owner}/{repo}/actions/runs?head_sha={head_sha}&per_page=100&page={page}",
            ]
        )
        if runs_result.returncode != 0:
            return PrOutcome(
                pr_number=pr_number,
                success=False,
                detail=f"run list failed: {runs_result.stderr.strip()}",
            )
        try:
            payload = json.loads(runs_result.stdout)
        except json.JSONDecodeError as exc:
            return PrOutcome(
                pr_number=pr_number,
                success=False,
                detail=f"could not parse run list: {exc}",
            )
        page_runs = payload.get("workflow_runs", [])
        if not isinstance(page_runs, list):
            return PrOutcome(
                pr_number=pr_number,
                success=False,
                detail="could not parse run list: workflow_runs is not a list",
            )
        if total_count is None:
            raw_total_count = payload.get("total_count", len(page_runs))
            total_count = (
                raw_total_count if isinstance(raw_total_count, int) else len(page_runs)
            )
        if not page_runs:
            break
        runs.extend(page_runs)
        page += 1

    rerunnable_conclusions = {"failure", "cancelled"}
    failed_run_ids = [
        r["id"]
        for r in runs
        if r.get("status") == "completed"
        and r.get("conclusion") in rerunnable_conclusions
    ]
    if not failed_run_ids:
        return PrOutcome(
            pr_number=pr_number,
            success=True,
            detail="no completed failed or cancelled runs to rerun",
        )

    reran: list[int] = []
    errors: list[str] = []
    for run_id in failed_run_ids:
        rerun_result = _run_gh(
            [
                "api",
                "-X",
                "POST",
                f"repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs",
            ]
        )
        if rerun_result.returncode == 0:
            reran.append(run_id)
        else:
            errors.append(f"run {run_id}: {rerun_result.stderr.strip()}")

    success = not errors
    detail = f"reran={reran}" + (f" errors={errors}" if errors else "")
    return PrOutcome(pr_number=pr_number, success=success, detail=detail)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_pr_numbers(raw: str) -> list[int]:
    numbers: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            numbers.append(int(chunk))
        except ValueError as exc:
            raise BulkPrThrottleError(f"invalid PR number {chunk!r} in --prs") from exc
    if not numbers:
        raise BulkPrThrottleError("--prs produced zero PR numbers")
    return numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mechanical dispatch throttle for bulk PR operations (OMN-16284)."
    )
    parser.add_argument(
        "--owner", required=True, help="GitHub org/owner. No default — fail-fast."
    )
    parser.add_argument(
        "--repo", required=True, help="GitHub repo name. No default — fail-fast."
    )
    parser.add_argument("--prs", required=True, help="Comma-separated PR numbers.")
    parser.add_argument("--operation", required=True, choices=VALID_OPERATIONS)
    parser.add_argument("--wave-size", type=int, default=DEFAULT_WAVE_SIZE)
    parser.add_argument(
        "--queue-depth-threshold", type=int, default=DEFAULT_QUEUE_DEPTH_THRESHOLD
    )
    parser.add_argument(
        "--max-total-prs",
        type=int,
        default=None,
        help=f"Explicitly raise the default cap ({DEFAULT_MAX_TOTAL_PRS}).",
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=DEFAULT_BLOCK_POLL_SECONDS
    )
    parser.add_argument(
        "--max-wait-seconds", type=float, default=DEFAULT_MAX_BLOCK_SECONDS
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Path to write the JSON wave receipt "
        "(default: .onex_state/bulk-pr-throttle/<owner>-<repo>-<ts>.json)",
    )
    args = parser.parse_args(argv)

    try:
        pr_numbers = _parse_pr_numbers(args.prs)
    except BulkPrThrottleError as exc:
        print(f"[bulk-pr-throttle] REFUSED: {exc}", file=sys.stderr)
        return 1

    explicit_max_total_prs = args.max_total_prs is not None
    max_total_prs = (
        args.max_total_prs if explicit_max_total_prs else DEFAULT_MAX_TOTAL_PRS
    )
    receipt_path = args.receipt
    if receipt_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        receipt_path = (
            Path(".onex_state")
            / "bulk-pr-throttle"
            / f"{args.owner}-{args.repo}-{ts}.json"
        )

    try:
        report = run_bulk_operation(
            owner=args.owner,
            repo=args.repo,
            pr_numbers=pr_numbers,
            operation=args.operation,
            wave_size=args.wave_size,
            queue_depth_threshold=args.queue_depth_threshold,
            max_total_prs=max_total_prs,
            explicit_max_total_prs=explicit_max_total_prs,
            dry_run=args.dry_run,
            get_queue_depth=(
                None if args.dry_run else lambda: gh_queue_depth(args.owner, args.repo)
            ),
            apply_pr_operation=(None if args.dry_run else gh_apply_pr_operation),
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
        )
    except PartialBulkRunError as exc:
        write_receipt(exc.report, receipt_path)
        print(f"[bulk-pr-throttle] receipt written to {receipt_path}")
        print(f"[bulk-pr-throttle] REFUSED: {exc}", file=sys.stderr)
        return 1
    except (BulkPrThrottleError, ValueError) as exc:
        print(f"[bulk-pr-throttle] REFUSED: {exc}", file=sys.stderr)
        return 1

    write_receipt(report, receipt_path)
    print(f"[bulk-pr-throttle] receipt written to {receipt_path}")

    failures = [o for wave in report.waves for o in wave.outcomes if not o.success]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

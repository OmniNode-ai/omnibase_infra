#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cancel superseded GitHub merge-queue workflow runs.

The janitor is intentionally conservative: it only considers ``merge_group``
runs, preserves the newest run for each live queue ref/workflow pair, and
defaults to dry-run mode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACTIVE_RUN_STATUSES = frozenset({"queued", "in_progress", "waiting", "pending"})


@dataclass(frozen=True)
class WorkflowRun:
    run_id: int
    event: str
    head_branch: str
    workflow_name: str
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CancellationCandidate:
    run_id: int
    head_branch: str
    workflow_name: str
    reason: str
    preserved_run_id: int | None = None


def determine_cancellations(
    *, live_queue_refs: set[str], active_runs: list[WorkflowRun]
) -> list[CancellationCandidate]:
    """Return merge-group runs that are safe to cancel."""
    candidates: list[CancellationCandidate] = []
    grouped: dict[tuple[str, str], list[WorkflowRun]] = {}

    for run in active_runs:
        if run.event != "merge_group" or run.status not in ACTIVE_RUN_STATUSES:
            continue
        if not _is_live_queue_ref(run.head_branch, live_queue_refs):
            candidates.append(
                CancellationCandidate(
                    run_id=run.run_id,
                    head_branch=run.head_branch,
                    workflow_name=run.workflow_name,
                    reason="stale_queue_ref",
                )
            )
            continue
        grouped.setdefault((run.head_branch, run.workflow_name), []).append(run)

    for (_head_branch, _workflow_name), runs in grouped.items():
        if len(runs) < 2:
            continue
        newest = max(runs, key=_run_sort_key)
        for run in runs:
            if run.run_id == newest.run_id:
                continue
            candidates.append(
                CancellationCandidate(
                    run_id=run.run_id,
                    head_branch=run.head_branch,
                    workflow_name=run.workflow_name,
                    reason="older_duplicate_workflow_for_live_ref",
                    preserved_run_id=newest.run_id,
                )
            )

    return sorted(candidates, key=lambda candidate: candidate.run_id)


def _is_live_queue_ref(head_branch: str, live_queue_refs: set[str]) -> bool:
    return head_branch in live_queue_refs or any(
        head_branch.startswith(live_ref) for live_ref in live_queue_refs
    )


def _run_sort_key(run: WorkflowRun) -> tuple[str, int]:
    return (run.created_at, run.run_id)


def _run_gh_json(args: list[str]) -> Any:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def fetch_live_queue_refs(owner: str, repo: str, branch: str) -> set[str]:
    query = """
query($owner: String!, $repo: String!, $branch: String!) {
  repository(owner: $owner, name: $repo) {
    mergeQueue(branch: $branch) {
      entries(first: 100) {
        nodes {
          pullRequest {
            number
          }
        }
      }
    }
  }
}
"""
    payload = _run_gh_json(
        [
            "api",
            "graphql",
            "-f",
            f"owner={owner}",
            "-f",
            f"repo={repo}",
            "-f",
            f"branch={branch}",
            "-f",
            f"query={query}",
        ]
    )
    nodes = payload["data"]["repository"]["mergeQueue"]["entries"]["nodes"]
    return {
        f"gh-readonly-queue/{branch}/pr-{node['pullRequest']['number']}-"
        for node in nodes
        if node.get("pullRequest")
    }


def fetch_active_runs(owner: str, repo: str, limit: int) -> list[WorkflowRun]:
    payload = _run_gh_json(
        [
            "run",
            "list",
            "--repo",
            f"{owner}/{repo}",
            "--limit",
            str(limit),
            "--json",
            "databaseId,event,headBranch,workflowName,status,createdAt,updatedAt",
        ]
    )
    return [
        WorkflowRun(
            run_id=int(item["databaseId"]),
            event=item["event"],
            head_branch=item["headBranch"],
            workflow_name=item["workflowName"],
            status=item["status"],
            created_at=item["createdAt"],
            updated_at=item["updatedAt"],
        )
        for item in payload
        if item["status"] in ACTIVE_RUN_STATUSES
    ]


def cancel_run(owner: str, repo: str, run_id: int) -> None:
    subprocess.run(
        ["gh", "run", "cancel", str(run_id), "--repo", f"{owner}/{repo}"],
        check=True,
    )


def write_report(
    *,
    path: Path,
    owner: str,
    repo: str,
    branch: str,
    apply: bool,
    live_queue_refs: set[str],
    candidates: list[CancellationCandidate],
) -> None:
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repository": f"{owner}/{repo}",
        "branch": branch,
        "mode": "apply" if apply else "dry-run",
        "live_queue_refs": sorted(live_queue_refs),
        "cancellations": [asdict(candidate) for candidate in candidates],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="OmniNode-ai")
    parser.add_argument("--repo", default="omnibase_infra")
    parser.add_argument("--branch", default="dev")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Cancel candidates. Omit for dry-run mode.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path.",
    )
    args = parser.parse_args(argv)

    live_queue_refs = fetch_live_queue_refs(args.owner, args.repo, args.branch)
    active_runs = fetch_active_runs(args.owner, args.repo, args.limit)
    candidates = determine_cancellations(
        live_queue_refs=live_queue_refs,
        active_runs=active_runs,
    )

    if not candidates:
        print("[merge-queue-janitor] no superseded merge_group runs found")
    for candidate in candidates:
        print(
            "[merge-queue-janitor] "
            f"{'cancel' if args.apply else 'would-cancel'} "
            f"run={candidate.run_id} workflow={candidate.workflow_name!r} "
            f"ref={candidate.head_branch!r} reason={candidate.reason}"
        )
        if args.apply:
            cancel_run(args.owner, args.repo, candidate.run_id)

    if args.report:
        write_report(
            path=args.report,
            owner=args.owner,
            repo=args.repo,
            branch=args.branch,
            apply=args.apply,
            live_queue_refs=live_queue_refs,
            candidates=candidates,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-candidate delivery liveness guard (OMN-16906).

Why a watcher, and not just the fix
-----------------------------------
The OMN-16906 outage was not hard to fix once seen — it was one YAML scalar.
It was hard to *see*. ``deliver-dev-candidate-to-staging.yml`` failed with
GitHub's ``startup_failure``, which creates no job, no step, and no annotation.
Nothing in either repo went red. The staging deploy that consumed the stale
image afterwards reported green, because from its side nothing was wrong: it
faithfully re-applied the pins it was given. The gap surfaced only when a
migration fence that had already merged failed to apply on onex-dev
(deploy run 33207733444), a full day later.

So the delivery chain has exactly two silent-failure shapes, and this guard
exists to make both of them page:

``STARTUP_FAILURE``
    The newest delivery run never compiled. This is the observed incident.

``NOT_FIRED``
    A ``dev`` commit that matches the workflow's own ``on.push.paths`` filter
    has no delivery run at all. This is the shape measured on 2026-08-28: dev
    commit ``7090f386f`` (PR #2974, the OMN-16493 migration fence) landed at
    20:13Z and produced *zero* runs of any conclusion. A guard that only looked
    at run conclusions would have reported clean through it.

``NOT_DELIVERED`` covers the ordinary case — the run exists, finished, and did
not succeed — which is at least visible in the Actions tab, but is worth the
same page since the consequence is identical: merged, not running.

Trigger-drift safety
--------------------
The path filter is read out of ``deliver-dev-candidate-to-staging.yml`` itself
rather than restated here. A guard carrying its own copy of the trigger would
go quietly wrong the moment the trigger changed — the same "two sources, one
truth" defect the delivery workflow's SINGLE_SOURCE_REV_BUNDLE note is about.

Fail-closed
-----------
Every indeterminate state is a FAIL, not a pass: an unreadable workflow file,
an empty run list, an API error, a path filter that matched nothing. A liveness
guard that reports green when it could not look is worse than no guard, because
it manufactures the exact false assurance that let this outage run for a day.

Exit codes: ``0`` delivery is live, ``1`` a delivery gap (or the guard could not
prove there was none).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # fixed argv, no shell, trusted gh binary
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

DELIVERY_WORKFLOW = "deliver-dev-candidate-to-staging.yml"
DEFAULT_REPO = "OmniNode-ai/omnibase_infra"
DEFAULT_BRANCH = "dev"

# A run RECORD is created within seconds of the push — even a startup_failure
# run appears immediately. So "no run exists" only becomes a finding after a
# short window that absorbs GitHub-side scheduling latency.
RUN_APPEARANCE_GRACE = timedelta(minutes=20)

# The candidate runtime build is allotted 75 minutes by its own
# `timeout-minutes`, plus queueing. A run still in flight inside this window is
# healthy, not late.
DELIVERY_COMPLETION_GRACE = timedelta(hours=3)


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str

    def render(self) -> str:
        return f"[{self.code}] {self.detail}"


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings


def _parse_ts(value: str) -> datetime:
    """Parse a GitHub ISO-8601 timestamp into an aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def path_filter(workflow_path: Path) -> list[str]:
    """Read ``on.push.paths`` out of the delivery workflow.

    Raises rather than defaulting: a guard that silently fell back to "match
    everything" or "match nothing" would be reporting on a trigger that is not
    the real one.
    """
    document = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    triggers = document.get("on", document.get(True))
    if not isinstance(triggers, dict):
        raise ValueError(f"{workflow_path.name} has no parseable `on:` block")
    push = triggers.get("push")
    if not isinstance(push, dict) or not isinstance(push.get("paths"), list):
        raise ValueError(
            f"{workflow_path.name} no longer declares `on.push.paths`; this guard "
            "reads the trigger from the workflow and cannot evaluate without it"
        )
    paths = [str(p) for p in push["paths"]]
    if not paths:
        raise ValueError(f"{workflow_path.name} declares an empty `on.push.paths`")
    return paths


def matches_filter(filename: str, patterns: list[str]) -> bool:
    """GitHub path-filter matching, restricted to the forms this trigger uses.

    The delivery trigger uses only two shapes — a literal file path and a
    ``dir/**`` prefix — so those are implemented exactly rather than
    approximated by a general glob translation that could differ from GitHub's
    semantics at the edges. Any other shape raises in :func:`evaluate` via
    :func:`assert_supported_patterns` rather than being silently mis-evaluated.
    """
    for pattern in patterns:
        if pattern.endswith("/**"):
            if filename.startswith(pattern[: -len("**")]):
                return True
        elif filename == pattern:
            return True
    return False


def assert_supported_patterns(patterns: list[str]) -> None:
    for pattern in patterns:
        if pattern.endswith("/**"):
            continue
        if any(ch in pattern for ch in "*?[]!"):
            raise ValueError(
                f"path filter {pattern!r} uses a glob shape this guard does not "
                "implement exactly; extend matches_filter() rather than letting "
                "the guard approximate the trigger"
            )


def evaluate(
    runs: list[dict[str, Any]],
    candidate_commits: list[dict[str, Any]],
    patterns: list[str],
    now: datetime,
) -> Verdict:
    """Decide whether dev-merge -> onex-dev delivery is live.

    ``runs`` are delivery-workflow runs on ``dev``, newest first.
    ``candidate_commits`` are recent ``dev`` commits, newest first, each with
    ``sha``, ``committed_at`` and the ``files`` it changed.
    """
    verdict = Verdict()
    assert_supported_patterns(patterns)

    if not runs:
        verdict.findings.append(
            Finding(
                "NO_RUNS",
                f"{DELIVERY_WORKFLOW} has no runs on {DEFAULT_BRANCH} at all — either the "
                "trigger was removed or the API returned nothing; refusing to "
                "report delivery healthy on an empty result.",
            )
        )
        return verdict

    newest = runs[0]
    if newest.get("conclusion") == "startup_failure":
        verdict.findings.append(
            Finding(
                "STARTUP_FAILURE",
                f"newest run {newest.get('id')} ({newest.get('created_at')}) is "
                "startup_failure: the workflow graph did not compile, so NO job ran "
                "and nothing else in either repo will go red. Every dev merge since "
                "is merged-but-undelivered. See "
                "docs/runbooks/dev-candidate-delivery-recovery.md.",
            )
        )

    matching = [
        commit
        for commit in candidate_commits
        if any(matches_filter(f, patterns) for f in commit.get("files", []))
    ]
    if not matching:
        verdict.notes.append(
            "no recent dev commit touched the delivery trigger paths; "
            "run-existence checks not applicable this cycle"
        )
        return verdict

    target = matching[0]
    committed_at = _parse_ts(target["committed_at"])
    age = now - committed_at
    runs_for_target = [r for r in runs if r.get("head_sha") == target["sha"]]

    if not runs_for_target:
        if age > RUN_APPEARANCE_GRACE:
            verdict.findings.append(
                Finding(
                    "NOT_FIRED",
                    f"dev commit {target['sha'][:9]} ({target['committed_at']}) changed "
                    f"delivery-trigger paths {age} ago and produced NO run of "
                    f"{DELIVERY_WORKFLOW}. The merge is not on its way to onex-dev and "
                    "no run record exists to notice.",
                )
            )
        else:
            verdict.notes.append(
                f"dev commit {target['sha'][:9]} is {age} old; still inside the "
                f"{RUN_APPEARANCE_GRACE} run-appearance grace"
            )
        return verdict

    run = runs_for_target[0]
    if run.get("conclusion") == "success":
        verdict.notes.append(
            f"dev commit {target['sha'][:9]} delivered by run {run.get('id')}"
        )
        return verdict

    if run.get("status") != "completed":
        if age <= DELIVERY_COMPLETION_GRACE:
            verdict.notes.append(
                f"run {run.get('id')} for {target['sha'][:9]} is {run.get('status')}, "
                f"inside the {DELIVERY_COMPLETION_GRACE} build window"
            )
            return verdict
        verdict.findings.append(
            Finding(
                "DELIVERY_STALLED",
                f"run {run.get('id')} for dev commit {target['sha'][:9]} has been "
                f"{run.get('status')} for {age}, past the "
                f"{DELIVERY_COMPLETION_GRACE} allowance.",
            )
        )
        return verdict

    verdict.findings.append(
        Finding(
            "NOT_DELIVERED",
            f"run {run.get('id')} for dev commit {target['sha'][:9]} completed with "
            f"conclusion={run.get('conclusion')!r}. onex-dev is not running this "
            "commit. See docs/runbooks/dev-candidate-delivery-recovery.md.",
        )
    )
    return verdict


def _gh(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def normalize_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Project one GitHub workflow-run object onto the fields the verdict uses.

    Split out from :func:`collect` so the incident replay can drive the real
    projection over the captured bytes of run 33169436998 rather than a
    hand-built dict. A replay that reconstructs the shape by hand proves the
    verdict logic and nothing about whether the guard can read what GitHub
    actually returns.
    """
    return {
        "id": payload["id"],
        "head_sha": payload["head_sha"],
        "status": payload["status"],
        "conclusion": payload["conclusion"],
        "created_at": payload["created_at"],
    }


def normalize_commit(payload: dict[str, Any]) -> dict[str, Any]:
    """Project one GitHub commit object onto the fields the verdict uses."""
    return {
        "sha": payload["sha"],
        "committed_at": payload["commit"]["committer"]["date"],
        "files": [f["filename"] for f in payload.get("files", [])],
    }


def collect(
    repo: str, branch: str, commit_window: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs_payload = _gh(
        [
            "api",
            f"repos/{repo}/actions/workflows/{DELIVERY_WORKFLOW}/runs"
            f"?branch={branch}&per_page=30",
        ]
    )
    runs = [normalize_run(r) for r in runs_payload.get("workflow_runs", [])]

    listing = _gh(
        ["api", f"repos/{repo}/commits?sha={branch}&per_page={commit_window}"]
    )
    commits = [
        normalize_commit(_gh(["api", f"repos/{repo}/commits/{entry['sha']}"]))
        for entry in listing
    ]
    return runs, commits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)
    )
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--workflow-file",
        type=Path,
        default=Path(".github/workflows") / DELIVERY_WORKFLOW,
    )
    parser.add_argument(
        "--commit-window",
        type=int,
        default=10,
        help="how many recent dev commits to inspect for trigger-path changes",
    )
    args = parser.parse_args(argv)

    try:
        patterns = path_filter(args.workflow_file)
        runs, commits = collect(args.repo, args.branch, args.commit_window)
        verdict = evaluate(runs, commits, patterns, datetime.now(UTC))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        # Fail closed: an unreadable trigger or an API failure is not evidence
        # that delivery is healthy.
        print(f"::error::delivery liveness guard could not evaluate: {exc}")
        return 1

    for note in verdict.notes:
        print(f"ok: {note}")
    if verdict.ok:
        print("dev-candidate delivery to onex-dev is live.")
        return 0

    for finding in verdict.findings:
        print(f"::error::{finding.render()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

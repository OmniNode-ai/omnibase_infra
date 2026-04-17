# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure-Python logic for audit-branch-protection.sh (OMN-9034).

Extracted from the shell script's inline python3 HEREDOCs so the logic can
be unit-tested by importing the functions directly. The shell wrapper calls
this module for Check A / Check B evaluation and fix-payload construction;
the tests exercise the same functions with synthetic gh-api JSON inputs.

See scripts/audit-branch-protection.sh for the orchestration layer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

GhCaller = Callable[[list[str]], tuple[int, str]]
"""A callable shaped like `(argv_without_gh) -> (returncode, stdout)`.

The production shell script passes a real `gh api`-invoking callable; tests
pass a fake that returns canned JSON. This is the injection seam that makes
the audit logic testable without spawning bash/gh as child processes.
"""

PAGE_SIZE = 100
"""GitHub REST pagination page size; 100 is the API maximum (OMN-9034 thread 7)."""


def parse_required_approving_review_count(protection_json: str) -> int:
    """Extract required_approving_review_count from a branch-protection API response.

    Returns 0 when the field is absent or the payload is malformed.
    """
    try:
        d = json.loads(protection_json)
    except json.JSONDecodeError:
        return 0
    prr = d.get("required_pull_request_reviews") or {}
    rac = prr.get("required_approving_review_count", 0)
    return int(rac) if isinstance(rac, int) else 0


def parse_required_contexts(protection_json: str) -> list[str]:
    """Extract required_status_checks.contexts from a branch-protection API response."""
    try:
        d = json.loads(protection_json)
    except json.JSONDecodeError:
        return []
    rsc = d.get("required_status_checks") or {}
    ctxs = rsc.get("contexts", [])
    return [str(c) for c in ctxs] if isinstance(ctxs, list) else []


def build_fix_payload(protection_json: str) -> dict[str, Any]:
    """Build the PUT payload used by --fix to drop required_pull_request_reviews.

    Preserves required_status_checks (contexts + strict) and enforce_admins,
    sets required_pull_request_reviews and restrictions to None. This mirrors
    the inline python3 block in audit-branch-protection.sh.
    """
    d = json.loads(protection_json)
    payload: dict[str, Any] = {}
    rsc = d.get("required_status_checks")
    if rsc:
        payload["required_status_checks"] = {
            "strict": rsc.get("strict", False),
            "contexts": rsc.get("contexts", []),
        }
    payload["enforce_admins"] = bool(
        (d.get("enforce_admins") or {}).get("enabled", False)
    )
    payload["required_pull_request_reviews"] = None
    payload["restrictions"] = None
    return payload


def collect_seen_check_run_names(
    owner: str,
    repo: str,
    commits: int,
    gh: GhCaller,
) -> set[str]:
    """Fetch check-run names across the last N commits on the default branch.

    Paginates check-runs per commit using PAGE_SIZE (thread 7 fix — was
    previously hardcoded to 50, missing contexts on busy repos).
    Returns the union of all check-run names seen.
    """
    rc, commits_stdout = gh(
        ["api", f"repos/{owner}/{repo}/commits?per_page={commits}", "--jq", ".[].sha"]
    )
    if rc != 0:
        return set()
    shas = [s for s in commits_stdout.splitlines() if s.strip()]

    seen: set[str] = set()
    for sha in shas:
        page = 1
        while True:
            rc, stdout = gh(
                [
                    "api",
                    f"repos/{owner}/{repo}/commits/{sha}/check-runs"
                    f"?per_page={PAGE_SIZE}&page={page}",
                ]
            )
            if rc != 0 or not stdout.strip():
                break
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                break
            runs = data.get("check_runs", [])
            if not runs:
                break
            for run in runs:
                name = run.get("name")
                if isinstance(name, str):
                    seen.add(name)
            if len(runs) < PAGE_SIZE:
                break
            page += 1
    return seen


def find_orphan_contexts(required: list[str], seen: set[str]) -> list[str]:
    """Return contexts that appear in `required` but not in `seen` check-run names."""
    return [c for c in required if c not in seen]


def audit_repo(
    owner: str,
    repo: str,
    gh: GhCaller,
    commits_to_scan: int = 5,
) -> dict[str, Any]:
    """Run Check A + Check B against a single repo.

    Returns a dict with:
      - status: "ok" | "skip" | "violation"
      - rac: int (required_approving_review_count, 0 if skipped)
      - required_contexts: list[str]
      - orphan_contexts: list[str]
      - message: human-readable summary
      - protection_json: raw protection payload (empty if skipped)
    """
    rc, protection = gh(["api", f"repos/{owner}/{repo}/branches/main/protection"])
    if rc != 0:
        return {
            "status": "skip",
            "rac": 0,
            "required_contexts": [],
            "orphan_contexts": [],
            "protection_json": "",
            "message": f"branch protection not enabled or inaccessible for {repo}",
        }

    rac = parse_required_approving_review_count(protection)
    required = parse_required_contexts(protection)
    seen = (
        collect_seen_check_run_names(owner, repo, commits_to_scan, gh)
        if required
        else set()
    )
    orphans = find_orphan_contexts(required, seen)

    violations: list[str] = []
    if rac > 0:
        violations.append(f"Check A: required_approving_review_count={rac} (must be 0)")
    for ctx in orphans:
        violations.append(
            f"Check B: context '{ctx}' not found in last-{commits_to_scan}-commit check-runs"
        )

    if violations:
        return {
            "status": "violation",
            "rac": rac,
            "required_contexts": required,
            "orphan_contexts": orphans,
            "protection_json": protection,
            "message": "; ".join(violations),
        }
    return {
        "status": "ok",
        "rac": rac,
        "required_contexts": required,
        "orphan_contexts": [],
        "protection_json": protection,
        "message": f"{repo}: clean",
    }

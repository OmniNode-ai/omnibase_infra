#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Orphaned required-status-check audit (OMN-9034 Check B, corrected by OMN-18346).

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
This is the ONE branch-protection check that the canonical auditor
(`onex_change_control/scripts/audit_branch_protection.sh`, run by this repo's
`branch-protection-audit.yml` at a pinned ref) does not carry: the ORPHANED
direction — a required context on `main` that no check-run actually reports any
more, so it can never go green and permanently blocks every PR targeting that
branch.

Everything else this repo used to audit locally — review enforcement, the
release-synced-main pair, `enforce_admins`, the Receipt Gate,
`delete_branch_on_merge`, the Merge Queue ruleset — is the canonical auditor's
job and is NOT duplicated here. Two copies of the same audit disagreeing is the
condition OMN-18346 exists to end: the in-repo copy carried no
`REVIEW_GATED_REPOS` carve-out, so it kept reporting `onex_change_control` as a
Check A violation for a code-owner requirement that is deliberate (OMN-18287).

THE EVIDENCE SOURCE IS THE WHOLE DEFECT (OMN-18346)
---------------------------------------------------
The previous implementation compared `main`'s `required_status_checks.contexts`
against check-runs on the last five commits of the DEFAULT branch. Those commits
are post-merge pushes, so they carry only the push-triggered subset. The
required contexts are PR-time gates that bind to a PR head SHA and can never
appear there, so the audit reported healthy repos as almost entirely orphaned
and failed 47 of 47 scheduled runs in seven days.

Measured on omniclaude at 2026-09-13 (positive control, live):

    required contexts on main .................. 56
    distinct check-run names on 5 dev pushes ... 12  -> 54 "orphans" (all false)
    distinct names incl. merged PR head SHAs ... 130 ->  2 orphans (both real)

The two that survive are real: no workflow in omniclaude emits
``CodeRabbit Thread Check`` any more, so those two required contexts block every
PR targeting omniclaude `main`. That is precisely the drift this check exists to
surface, and it is the reason the check was not simply retired.

So the evidence source here is the union of:
  * check-runs on recent DEFAULT-BRANCH commits (push-triggered gates), and
  * check-runs on the head SHAs of recently MERGED PULL REQUESTS (PR-time gates).

MATCHING MIRRORS GITHUB, NOT STRING EQUALITY
---------------------------------------------
GitHub satisfies a reusable-workflow required context of the form
``"caller-job / reusable-job"`` with the LEAF check-run name it actually
reports. A required ``"call-reject-skip-token / scan / reject-skip-gate-token"``
is satisfied by a reported ``"reject-skip-gate-token"``. Comparing raw strings
would report every such context as an orphan even though GitHub considers it
green, so both sides are compared under leaf normalization. The consequence is
stated rather than hidden: two required contexts that share a leaf are
indistinguishable here, exactly as they are to GitHub's own matcher.

NO ALLOWLIST (OMN-18346 removed the last one)
----------------------------------------------
The old implementation carried ``PR_ONLY_CONTEXTS = {"main-target-guard"}``
because that context binds to a PR head SHA and the push-only evidence source
could never observe it. Under the corrected evidence source it IS observed —
live control on omniclaude, 2026-09-13: present on 22 of the 26 merged PR head
SHAs in the sample window. The allowlist was a workaround for the defect this
module fixes, so it is gone rather than inherited: an allowlisted context is a
context whose genuine disappearance can no longer be reported.

AN EMPTY EVIDENCE SET IS NOT EVIDENCE OF ABSENCE
-------------------------------------------------
If a repo declares required contexts but NO check-run name could be collected at
all, this module reports ``indeterminate`` and judges nothing. A zero-row
evidence read and a genuinely orphaned repo look identical otherwise, and
treating the first as the second is how the original defect manifested at
47-runs-of-47 scale. Requiring a non-empty evidence set before any orphan is
reported is the positive control, built in.

READ-ONLY. This module performs no mutation of any kind — there is no ``--fix``
path. The remediation for an orphan is a branch-protection change made
deliberately by an operator, not by a scheduled audit.

Usage:
    python3 scripts/audit_orphan_required_contexts.py --owner OmniNode-ai
    python3 scripts/audit_orphan_required_contexts.py --repo omniclaude

Exit codes:
    0 — no orphaned contexts found
    1 — at least one orphaned required context found
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

GhCaller = Callable[[list[str]], tuple[int, str]]
"""A callable shaped like ``(argv_without_gh) -> (returncode, stdout)``.

Production passes a real ``gh api``-invoking callable; tests pass a fake that
serves captured bytes. This is the injection seam that lets the incident replay
drive the REAL audit logic over the REAL captured API responses.
"""

PAGE_SIZE = 100
"""GitHub REST pagination page size; 100 is the API maximum."""

DEFAULT_PUSH_COMMITS = 5
"""Default-branch commits sampled for push-triggered check-run evidence."""

DEFAULT_MERGED_PRS = 30
"""Closed PRs sampled (merged ones kept) for PR-time check-run evidence.

Sized against the live control: omniclaude's 56 required contexts are all
observed within this window, and a 30-PR window on this fleet spans roughly a
week — long enough that an ordinary quiet weekend cannot manufacture an orphan.
"""

REPOS: tuple[str, ...] = (
    "omniclaude",
    "omnibase_core",
    "omnibase_infra",
    "omnibase_spi",
    "omnidash",
    "omniintelligence",
    "omnimemory",
    "omninode_infra",
    "omniweb",
    "onex_change_control",
)


def leaf_form(context: str) -> str:
    """Return the trailing ``" / "``-separated segment of a context name.

    Mirrors GitHub's reusable-workflow context matching: a required
    ``"deploy-gate / deploy-gate"`` is satisfied by a reported leaf
    ``"deploy-gate"``.
    """
    return context.split(" / ")[-1].strip()


def normalize_forms(context: str) -> set[str]:
    """Return every form under which ``context`` may legitimately be observed."""
    return {context.strip(), leaf_form(context)}


def parse_required_contexts(protection_json: str) -> list[str]:
    """Extract required status-check contexts from a branch-protection payload.

    Reads BOTH the legacy ``contexts[]`` array and the newer ``checks[].context``
    form, because a repo can carry either and a reader that knows only one
    silently audits nothing on the other.
    """
    try:
        payload = json.loads(protection_json)
    except json.JSONDecodeError:
        return []
    rsc = payload.get("required_status_checks") or {}
    names: list[str] = []
    raw = rsc.get("contexts")
    if isinstance(raw, list):
        names.extend(str(c) for c in raw)
    checks = rsc.get("checks")
    if isinstance(checks, list):
        names.extend(str(c.get("context")) for c in checks if isinstance(c, dict))
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name and name != "None" and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def collect_default_branch_shas(
    owner: str, repo: str, count: int, gh: GhCaller
) -> list[str]:
    """Recent DEFAULT-BRANCH commit SHAs — the push-triggered evidence source."""
    rc, stdout = gh(["api", f"repos/{owner}/{repo}/commits?per_page={count}"])
    if rc != 0 or not stdout.strip():
        return []
    try:
        commits = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(commits, list):
        return []
    return [str(c["sha"]) for c in commits if isinstance(c, dict) and c.get("sha")]


def collect_merged_pr_head_shas(
    owner: str, repo: str, count: int, gh: GhCaller
) -> list[str]:
    """Head SHAs of recently MERGED pull requests — the PR-time evidence source.

    This is the source the previous implementation lacked entirely, and its
    absence is the whole of OMN-18346. A PR-time required context binds its
    check-run to the PR head SHA, so this is the only surface on which such a
    context is observable at all.
    """
    rc, stdout = gh(
        [
            "api",
            f"repos/{owner}/{repo}/pulls"
            f"?state=closed&per_page={count}&sort=updated&direction=desc",
        ]
    )
    if rc != 0 or not stdout.strip():
        return []
    try:
        pulls = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(pulls, list):
        return []
    shas: list[str] = []
    for pull in pulls:
        if not isinstance(pull, dict) or not pull.get("merged_at"):
            continue
        head = pull.get("head") or {}
        sha = head.get("sha") if isinstance(head, dict) else None
        if sha:
            shas.append(str(sha))
    return shas


def collect_check_run_names(
    owner: str, repo: str, shas: list[str], gh: GhCaller
) -> set[str]:
    """Union of check-run names across ``shas``, paginated at the API maximum."""
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
            runs = data.get("check_runs") or []
            if not runs:
                break
            for run in runs:
                name = run.get("name")
                if isinstance(name, str) and name:
                    seen.add(name)
            if len(runs) < PAGE_SIZE:
                break
            page += 1
    return seen


def find_orphan_contexts(required: list[str], seen: set[str]) -> list[str]:
    """Required contexts that no observed check-run name can satisfy.

    Both sides are compared under leaf normalization so the result matches what
    GitHub itself would consider satisfied (see the module docstring).
    """
    observed: set[str] = set()
    for name in seen:
        observed |= normalize_forms(name)
    return [c for c in required if not (normalize_forms(c) & observed)]


def audit_repo_main(
    owner: str,
    repo: str,
    gh: GhCaller,
    push_commits: int = DEFAULT_PUSH_COMMITS,
    merged_prs: int = DEFAULT_MERGED_PRS,
) -> dict[str, Any]:
    """Audit ONE repo's `main` for orphaned required contexts. READ-ONLY.

    Returns a dict with ``status`` in ``{"ok", "skip", "indeterminate",
    "violation"}``, the ``required_contexts`` and ``orphan_contexts`` lists, the
    size of the evidence set, and a human-readable ``message``.

    ``indeterminate`` is returned — never ``violation`` — when required contexts
    exist but no check-run name could be collected at all. See the module
    docstring: an unresolvable evidence read is not a finding about branch
    protection, and reporting it as one is the original defect.
    """
    rc, protection = gh(["api", f"repos/{owner}/{repo}/branches/main/protection"])
    if rc != 0:
        return {
            "status": "skip",
            "required_contexts": [],
            "orphan_contexts": [],
            "evidence_size": 0,
            "message": f"branch protection not enabled or inaccessible for {repo} (main)",
        }

    required = parse_required_contexts(protection)
    if not required:
        return {
            "status": "ok",
            "required_contexts": [],
            "orphan_contexts": [],
            "evidence_size": 0,
            "message": (
                f"{repo} (main): no required status checks to orphan "
                "(expected on a release-synced main)"
            ),
        }

    shas = collect_default_branch_shas(owner, repo, push_commits, gh)
    shas += collect_merged_pr_head_shas(owner, repo, merged_prs, gh)
    seen = collect_check_run_names(owner, repo, shas, gh)

    # Positive control, built in: refuse to judge on a zero-row evidence read.
    if not seen:
        return {
            "status": "indeterminate",
            "required_contexts": required,
            "orphan_contexts": [],
            "evidence_size": 0,
            "message": (
                f"{repo} (main): {len(required)} required context(s) but ZERO check-run "
                f"names collected across {len(shas)} sampled SHA(s) — evidence source "
                "unresolved, nothing judged (an empty read is not evidence of absence)"
            ),
        }

    orphans = find_orphan_contexts(required, seen)
    if orphans:
        return {
            "status": "violation",
            "required_contexts": required,
            "orphan_contexts": orphans,
            "evidence_size": len(seen),
            "message": "; ".join(
                f"required context '{c}' matches no check-run observed across "
                f"{len(shas)} recent default-branch commits and merged PR head SHAs "
                f"({len(seen)} distinct names) — it can never report, so it blocks "
                "every PR targeting main"
                for c in orphans
            ),
        }
    return {
        "status": "ok",
        "required_contexts": required,
        "orphan_contexts": [],
        "evidence_size": len(seen),
        "message": (
            f"{repo} (main): all {len(required)} required context(s) observed "
            f"across {len(shas)} SHA(s) / {len(seen)} distinct check-run names"
        ),
    }


def _real_gh(argv: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["gh", *argv], capture_output=True, text=True, timeout=60, check=False
    )
    return proc.returncode, proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit orphaned required status-check contexts on `main`."
    )
    parser.add_argument("--owner", default="OmniNode-ai")
    parser.add_argument("--repo", default=None, help="Audit a single repo")
    parser.add_argument("--push-commits", type=int, default=DEFAULT_PUSH_COMMITS)
    parser.add_argument("--merged-prs", type=int, default=DEFAULT_MERGED_PRS)
    args = parser.parse_args(argv)

    repos = (args.repo,) if args.repo else REPOS
    print(
        f"=== orphaned required-context audit (main): owner={args.owner}, "
        f"repos={len(repos)}, evidence={args.push_commits} default-branch commits "
        f"+ up to {args.merged_prs} merged PR head SHAs ==="
    )
    print("")

    violations = 0
    for repo in repos:
        result = audit_repo_main(
            args.owner, repo, _real_gh, args.push_commits, args.merged_prs
        )
        status = result["status"]
        if status == "violation":
            violations += len(result["orphan_contexts"])
            print(f"--- {repo}")
            for ctx in result["orphan_contexts"]:
                print(f"    [main] [FAIL] orphaned required context: '{ctx}'")
            print(
                f"    [main] evidence: {result['evidence_size']} distinct check-run "
                f"names; {len(result['required_contexts'])} required context(s)"
            )
        elif status == "indeterminate":
            print(f"--- {repo}")
            print(f"    [main] [INDETERMINATE] {result['message']}")
        elif status == "skip":
            print(f"--- {repo}")
            print(f"    [main] [SKIP] {result['message']}")
        else:
            print(f"--- {repo}")
            print(f"    [main] [OK]   {result['message']}")

    print("")
    if violations == 0:
        print("PASS: no orphaned required contexts found.")
        return 0
    print(f"FAIL: {violations} orphaned required context(s) found.")
    print(
        "Remediation is an operator branch-protection change removing the stale "
        "context; this audit never mutates."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

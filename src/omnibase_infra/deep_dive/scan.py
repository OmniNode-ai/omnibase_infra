# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Git/``gh`` scan helpers for deep-dive report generation (OMN-13725).

This module is the *single source of truth* for the deterministic
git-and-``gh`` scan logic used by both the ``generate_deep_dive.py`` script
and the ``node_deep_dive_report_effect`` data-source adapter.  It performs
local subprocess I/O (``git`` / ``gh``) and parses the output into the typed
models defined in :mod:`omnibase_infra.deep_dive.models`.

Pure transformation/classification/scoring logic lives in
:mod:`omnibase_infra.deep_dive.processing`; this module imports those helpers
so there is no logic fork (R7).  Subprocess execution lives here only — it is
the legitimate I/O boundary and must never be invoked from a node *handler*
body (handlers route through an injected adapter that calls these helpers).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

from omnibase_infra.deep_dive.models import (
    ActiveWorktree,
    CommitEntry,
    DriftReport,
    GitHubMergedPR,
    MergeEntry,
    RepoDay,
)
from omnibase_infra.deep_dive.processing import (
    classify_pr,
    is_exempt_pr,
    is_workflow_pr,
)


def run_git(cmd: list[str], cwd: Path, *, allow_fail: bool = False) -> str:
    """Run a git/``gh`` subprocess and return its stdout.

    Returns an empty string when *allow_fail* is True and the command exits
    non-zero (mirrors the original script's ``_run`` semantics).
    """
    try:
        return subprocess.check_output(
            cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        if allow_fail:
            return ""
        raise


def get_branch(repo: Path) -> str:
    b = run_git(["git", "branch", "--show-current"], repo, allow_fail=True).strip()
    return b or "(detached)"


def get_dirty(repo: Path) -> list[str]:
    raw = run_git(["git", "status", "--porcelain=v1"], repo, allow_fail=True)
    return [line for line in raw.splitlines() if line.strip()]


def get_commit_entries(repo: Path, start_s: str, end_s: str) -> list[CommitEntry]:
    # We intentionally include merge commits in the main log; merge list is separate for convenience.
    raw = run_git(
        [
            "git",
            "log",
            f"--since={start_s}",
            f"--until={end_s}",
            "--pretty=format:%H|%h|%ai|%an|%s",
            "--shortstat",
        ],
        repo,
        allow_fail=True,
    )
    lines = raw.splitlines()
    entries: list[CommitEntry] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if "|" not in line:
            continue
        full, short, ai, an, subj = line.split("|", 4)
        files = ins = dele = 0
        while i < len(lines) and lines[i].strip() and "|" not in lines[i]:
            st = lines[i]
            i += 1
            m = re.search(r"(\d+) files? changed", st)
            if m:
                files += int(m.group(1))
            m = re.search(r"(\d+) insertions?\(\+\)", st)
            if m:
                ins += int(m.group(1))
            m = re.search(r"(\d+) deletions?\(-\)", st)
            if m:
                dele += int(m.group(1))
        entries.append(
            CommitEntry(
                full=full,
                short=short,
                ai=ai,
                author=an,
                subject=subj,
                files=files,
                ins=ins,
                dele=dele,
            )
        )
    return entries


def get_merge_entries(repo: Path, start_s: str, end_s: str) -> list[MergeEntry]:
    raw = run_git(
        [
            "git",
            "log",
            f"--since={start_s}",
            f"--until={end_s}",
            "--pretty=format:%H|%h|%ai|%an|%s",
            "--merges",
        ],
        repo,
        allow_fail=True,
    )
    merges: list[MergeEntry] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        full, short, ai, an, subj = line.split("|", 4)
        merges.append(
            MergeEntry(full=full, short=short, ai=ai, author=an, subject=subj)
        )
    return merges


def get_github_merged_prs(repo: Path, date: dt.date) -> list[GitHubMergedPR]:
    """
    Fetch PRs merged on the given date from GitHub using the gh CLI.

    Timestamps are converted from UTC to EST (America/New_York) before filtering,
    so PRs merged in the evening EST show up in the correct day's report.

    Returns an empty list if:
    - gh CLI is not available
    - The repo is not a GitHub repo
    - Any error occurs (network, auth, etc.)
    """
    date_str = date.isoformat()
    est_tz = ZoneInfo("America/New_York")

    raw = run_git(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "merged",
            "--search",
            f"merged:>={date_str}",
            "--json",
            "number,title,mergedAt,additions,deletions",
            "--limit",
            "100",
        ],
        repo,
        allow_fail=True,
    )
    if not raw.strip():
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    prs: list[GitHubMergedPR] = []
    for item in data:
        merged_at_str = item.get("mergedAt", "")
        title = item.get("title", "")

        if not merged_at_str:
            continue

        # Parse UTC timestamp and convert to EST
        try:
            # GitHub format: 2026-01-23T00:19:09Z
            merged_at_utc = dt.datetime.fromisoformat(
                merged_at_str.replace("Z", "+00:00")
            )
            merged_at_est = merged_at_utc.astimezone(est_tz)
            merged_date_est = merged_at_est.date()

            # Filter to only PRs merged on the target date (in EST)
            if merged_date_est == date:
                # Format display time in EST (table adds timezone label)
                display_time = merged_at_est.strftime("%H:%M")
                prs.append(
                    GitHubMergedPR(
                        number=item.get("number", 0),
                        title=title,
                        merged_at=display_time,
                        is_workflow_pr=is_workflow_pr(title),
                        category=classify_pr(title),
                        is_exempt=is_exempt_pr(title),
                        additions=item.get("additions", 0),
                        deletions=item.get("deletions", 0),
                    )
                )
        except (ValueError, TypeError):
            # If parsing fails, fall back to string prefix matching
            if merged_at_str.startswith(date_str):
                prs.append(
                    GitHubMergedPR(
                        number=item.get("number", 0),
                        title=title,
                        merged_at=merged_at_str,
                        is_workflow_pr=is_workflow_pr(title),
                        category=classify_pr(title),
                        is_exempt=is_exempt_pr(title),
                        additions=item.get("additions", 0),
                        deletions=item.get("deletions", 0),
                    )
                )

    # Sort by merge time
    return sorted(prs, key=lambda p: p.merged_at)


def find_git_repos_direct_children(root: Path) -> list[Path]:
    repos: list[Path] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        if (p / ".git").exists():
            repos.append(p)
    return repos


def compute_drift(repo_days: list[RepoDay], root: Path, date: dt.date) -> DriftReport:
    """
    Compute drift score based on actual risk signals, not parallelism.

    Only scans repos with activity today (commits or merged PRs).  Legacy
    repos with no today-activity are ignored — they're not part of the
    current work surface.

    Dirty worktrees on feature branches are NORMAL — they represent active
    parallel work sessions. Drift is measured by:
    1. Dirty files on main/master (uncommitted changes on integration branch)
    2. Stale feature branches (no commits in >72h — forgotten work)
    3. Diverged branches (feature branch behind origin/main by >48h)
    """
    main_dirty = 0
    stale_branches = 0
    diverged_branches = 0
    active_worktrees = 0
    risks: list[tuple[str, str]] = []

    now = dt.datetime.combine(date, dt.time(23, 59, 59)).astimezone()

    # Only consider repos with actual activity (commits or merged PRs), not
    # repos included solely because of --include-dirty.  Legacy repos with
    # dirty files but no today-activity are not part of the work surface.
    active_repo_days = [rd for rd in repo_days if rd.commits or rd.github_merged_prs]

    for rd in active_repo_days:
        name = rd.name
        repo_path = rd.path
        branch = rd.branch
        dirty = rd.dirty

        if dirty:
            active_worktrees += 1

        # Risk 1: Dirty files on main/master (real risk — uncommitted on integration branch)
        if dirty and branch in ("main", "master"):
            main_dirty += 1
            risks.append((name, f"{len(dirty)} dirty files on {branch}"))

        # Risk 2 & 3: Feature branch staleness and divergence
        if branch not in ("main", "master", "(detached)"):
            try:
                # Check last commit age on this branch
                last_commit_date = run_git(
                    ["git", "log", "-1", "--format=%ai", branch],
                    repo_path,
                    allow_fail=True,
                ).strip()
                if last_commit_date:
                    try:
                        lc_dt = dt.datetime.fromisoformat(last_commit_date.strip())
                        age = now - lc_dt.astimezone()
                        if age.total_seconds() > 72 * 3600:
                            stale_branches += 1
                            days = int(age.total_seconds() / 86400)
                            risks.append(
                                (name, f"branch `{branch}` stale ({days}d, no commits)")
                            )
                    except (ValueError, TypeError):
                        pass

                # Check if branch has diverged from origin/main
                merge_base_date = run_git(
                    ["git", "log", "-1", "--format=%ai", f"origin/main..{branch}"],
                    repo_path,
                    allow_fail=True,
                ).strip()
                if merge_base_date:
                    try:
                        mb_dt = dt.datetime.fromisoformat(merge_base_date.strip())
                        age = now - mb_dt.astimezone()
                        if age.total_seconds() > 48 * 3600:
                            diverged_branches += 1
                    except (ValueError, TypeError):
                        pass
            except Exception:  # noqa: BLE001 — boundary: swallows for resilience
                pass

    # Scoring: only real risk signals count
    # main_dirty is the strongest signal (uncommitted on integration branch)
    # stale_branches is moderate (forgotten work that will conflict)
    # diverged_branches is informational (will need rebase, but that's normal)
    risk_score = main_dirty * 3 + stale_branches * 2 + max(0, diverged_branches - 2)

    if risk_score >= 5:
        level = "red"
        penalty = -5
    elif risk_score >= 2:
        level = "yellow"
        penalty = -2
    else:
        level = "green"
        penalty = 0

    # Sort risks by severity (main dirty first, then stale, then other)
    risks.sort(
        key=lambda r: (
            0 if "dirty files on main" in r[1] else 1 if "stale" in r[1] else 2
        )
    )

    # Count unlinked PRs (no OMN-XXXX and not exempt/workflow)
    _ticket_re = re.compile(r"OMN-\d+")
    all_prs = [pr for rd in active_repo_days for pr in rd.github_merged_prs]
    unlinked = [
        pr
        for pr in all_prs
        if not pr.is_exempt
        and not pr.is_workflow_pr
        and not _ticket_re.search(pr.title)
    ]

    return DriftReport(
        level=level,
        main_dirty=main_dirty,
        stale_branches=stale_branches,
        diverged_branches=diverged_branches,
        risks=risks[:5],
        penalty=penalty,
        active_worktrees=active_worktrees,
        unlinked_pr_count=len(unlinked),
        total_pr_count=len(all_prs),
    )


def get_active_worktrees(repo: Path, repo_name: str) -> list[ActiveWorktree]:
    """
    Return non-main worktrees for a repo via `git worktree list --porcelain`.

    The main worktree (the bare clone or the primary checkout) is excluded
    since it already appears in the repo-day commit log.  Only feature-branch
    worktrees are returned — these represent in-progress work that may not
    have any commits today and would otherwise be invisible in the report.

    Returns an empty list on any error (allow_fail semantics).
    """
    raw = run_git(["git", "worktree", "list", "--porcelain"], repo, allow_fail=True)
    if not raw.strip():
        return []

    worktrees: list[ActiveWorktree] = []
    current: dict[str, str] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            # End of a worktree block
            if current:
                path = current.get("worktree", "")
                branch_ref = current.get("branch", "")
                head = current.get("HEAD", "")[:8] or "(unknown)"
                is_bare = "bare" in current

                # Derive a friendly branch name from the ref
                if is_bare:
                    branch = "(bare)"
                elif branch_ref.startswith("refs/heads/"):
                    branch = branch_ref[len("refs/heads/") :]
                else:
                    branch = branch_ref or "(detached)"

                # Skip main/master and bare entries — they're in the commit log already
                if branch not in ("main", "master", "(bare)") and path:
                    worktrees.append(
                        ActiveWorktree(
                            repo_name=repo_name,
                            worktree_path=path,
                            branch=branch,
                            head=head,
                        )
                    )
                current = {}
        elif ":" in line:
            key, _, val = line.partition(" ")
            current[key] = val.strip()
        else:
            # Lines like "bare" or "detached" are flags
            current[line] = line

    # Handle final block if no trailing blank line
    if current:
        path = current.get("worktree", "")
        branch_ref = current.get("branch", "")
        head = current.get("HEAD", "")[:8] or "(unknown)"
        is_bare = "bare" in current
        if is_bare:
            branch = "(bare)"
        elif branch_ref.startswith("refs/heads/"):
            branch = branch_ref[len("refs/heads/") :]
        else:
            branch = branch_ref or "(detached)"
        if branch not in ("main", "master", "(bare)") and path:
            worktrees.append(
                ActiveWorktree(
                    repo_name=repo_name,
                    worktree_path=path,
                    branch=branch,
                    head=head,
                )
            )

    return worktrees

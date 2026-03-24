#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Generate a daily deep dive Markdown report by scanning git repositories under a root directory.

Design goals:
- Deterministic output given the same repo state + date.
- No network required (uses only local git metadata).
- Focused coverage: include only git repos that have commits on the specified date.
- Stale dirty working trees (no commits today) are excluded by default to reduce noise.

Usage:
  python3 scripts/generate_deep_dive.py
  python3 scripts/generate_deep_dive.py --date 2025-12-20
  python3 scripts/generate_deep_dive.py --root /Volumes/PRO-G40/Code/omni_home --out /tmp/DECEMBER_20_2025_DEEP_DIVE.md
  python3 scripts/generate_deep_dive.py --include-dirty  # Also include repos with only dirty files (no commits)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo


def _run(cmd: list[str], cwd: Path, *, allow_fail: bool = False) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        if allow_fail:
            return ""
        raise


def _today_local() -> dt.date:
    return dt.datetime.now().astimezone().date()


def _day_window(date: dt.date) -> tuple[str, str]:
    # Local time window, consistent with most deep-dive narratives.
    start = dt.datetime.combine(date, dt.time(0, 0, 0))
    end = dt.datetime.combine(date, dt.time(23, 59, 59))
    return (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))


PR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(#(?P<num>\d+)\)"),
    re.compile(r"\bPR\s*#(?P<num>\d+)\b", re.IGNORECASE),
    re.compile(r"\b#(?P<num>\d+)\b"),  # fallback; can overmatch, but helps in practice
    re.compile(r"Merge pull request #(?P<num>\d+)", re.IGNORECASE),
]


def extract_pr_numbers(subject: str) -> list[int]:
    prs: set[int] = set()
    for pat in PR_PATTERNS:
        for m in pat.finditer(subject):
            try:
                prs.add(int(m.group("num")))
            except Exception:  # noqa: BLE001 — boundary: returns degraded response
                # Defensive: don't let parsing crash the run.
                continue
    return sorted(prs)


TICKET_RE = re.compile(r"\b(OMN-\d+)\b")


def extract_ticket_ids(subject: str) -> list[str]:
    return sorted(set(TICKET_RE.findall(subject)))


def collect_all_ticket_ids(repo_days: list[RepoDay]) -> list[str]:
    """Extract all unique ticket IDs (matching the TICKET_RE pattern) from commit messages."""
    ids: set[str] = set()
    for rd in repo_days:
        for c in rd.commits:
            ids.update(extract_ticket_ids(c.subject))
        for m in rd.merges:
            ids.update(extract_ticket_ids(m.subject))
    return sorted(ids, key=lambda x: int(x.split("-")[1]))


@dataclass(frozen=True)
class CommitEntry:
    full: str
    short: str
    ai: str
    author: str
    subject: str
    files: int
    ins: int
    dele: int


@dataclass(frozen=True)
class MergeEntry:
    full: str
    short: str
    ai: str
    author: str
    subject: str


@dataclass(frozen=True)
class GitHubMergedPR:
    number: int
    title: str
    merged_at: str
    is_workflow_pr: bool  # True if it's an auto-generated workflow PR
    category: str  # 'capability', 'correctness', 'governance', 'observability', 'docs', 'churn'
    additions: int = 0
    deletions: int = 0


@dataclass(frozen=True)
class RepoDay:
    name: str
    path: Path
    branch: str
    commits: list[CommitEntry]
    merges: list[MergeEntry]
    dirty: list[str]
    github_merged_prs: list[GitHubMergedPR]


def get_branch(repo: Path) -> str:
    b = _run(["git", "branch", "--show-current"], repo, allow_fail=True).strip()
    return b or "(detached)"


def get_dirty(repo: Path) -> list[str]:
    raw = _run(["git", "status", "--porcelain=v1"], repo, allow_fail=True)
    return [line for line in raw.splitlines() if line.strip()]


def get_commit_entries(repo: Path, start_s: str, end_s: str) -> list[CommitEntry]:
    # We intentionally include merge commits in the main log; merge list is separate for convenience.
    raw = _run(
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
    raw = _run(
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


WORKFLOW_PR_PATTERNS = [
    "Add Claude Code GitHub Workflow",
    "Update Claude Code Review workflow",
    "Update Claude PR Assistant workflow",
]


def is_workflow_pr(title: str) -> bool:
    """Check if a PR is an auto-generated workflow PR (low value for reporting)."""
    return any(pattern.lower() in title.lower() for pattern in WORKFLOW_PR_PATTERNS)


def classify_pr(title: str) -> str:
    """
    Classify a PR into exactly one of six categories using deterministic
    title-based heuristics with override rules.

    Categories: capability, correctness, governance, observability, docs, churn
    """
    t = title.lower()

    # Override rules (checked first -- these override prefix-based classification)
    if any(
        kw in t
        for kw in [
            "correct report",
            "report accuracy",
            "revert",
            "follow-up fix",
            "followup fix",
        ]
    ):
        return "churn"
    if any(
        kw in t
        for kw in [
            "handshake",
            "freeze",
            "enforcement",
            "policy gate",
            "migration_freeze",
        ]
    ):
        return "governance"
    if any(
        kw in t
        for kw in [
            "diagnostics",
            "telemetry",
            "metrics",
            "sink",
            "query reader",
            "projection",
            "ledger",
            "bus audit",
            "bus health",
        ]
    ):
        return "observability"

    # Prefix-based classification (conventional commits)
    if t.startswith(("docs", "doc(", "doc:")):
        return "docs"
    if t.startswith(("ci", "chore(ci)")) or "ci:" in t:
        return "governance"
    if t.startswith(("fix", "refactor", "perf", "test")):
        return "correctness"
    if t.startswith("feat"):
        return "capability"
    if t.startswith("chore"):
        return "correctness"

    # Default
    return "correctness"


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

    raw = _run(
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


_FAMILY_RE = re.compile(r"^(omni\w+?)(\d+)$")


def family_key(repo_name: str) -> str | None:
    """
    Map repo clones to their canonical family name for deduplication.

    Uses regex to strip trailing digits: omnibase_core2 -> omnibase_core, omnidash4 -> omnidash.
    Returns None for repos without trailing digits (they are already canonical).
    """
    m = _FAMILY_RE.match(repo_name)
    if m:
        return m.group(1)
    # Check if it's already a known omni-family base name
    if repo_name.startswith("omni"):
        return repo_name
    return None


@dataclass(frozen=True)
class DriftReport:
    level: str  # "green", "yellow", "red"
    main_dirty: int  # clones on main with uncommitted changes (real risk)
    stale_branches: int  # feature branches with last commit >72h ago
    diverged_branches: int  # feature branches diverged from origin/main >48h
    risks: list[tuple[str, str]]  # (repo_name, reason)
    penalty: int  # 0, -2, or -5
    # Informational (not scored)
    active_worktrees: int  # total dirty worktrees (parallelism, not risk)


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
                last_commit_date = _run(
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
                merge_base_date = _run(
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

    return DriftReport(
        level=level,
        main_dirty=main_dirty,
        stale_branches=stale_branches,
        diverged_branches=diverged_branches,
        risks=risks[:5],
        penalty=penalty,
        active_worktrees=active_worktrees,
    )


_CATEGORY_WEIGHTS: dict[str, float] = {
    "capability": 2.0,
    "correctness": 1.5,
    "governance": 1.5,
    "observability": 1.2,
    "docs": 0.7,
    "churn": 0.2,
}


def effectiveness_score_v2(
    category_counts: dict[str, int],
    drift_penalty: int,
) -> tuple[int, str]:
    """
    Compute effectiveness score based on weighted PR categories.

    Returns (score, explanation_string).
    """
    base = 60
    pr_points = 0.0
    total_prs = sum(category_counts.values())

    for cat, count in category_counts.items():
        pr_points += _CATEGORY_WEIGHTS.get(cat, 0.5) * count

    pr_points = min(pr_points, 50.0)

    penalty = 0
    churn_count = category_counts.get("churn", 0)
    churn_ratio = churn_count / max(total_prs, 1)
    if churn_ratio > 0.20:
        penalty += 3

    penalty += abs(drift_penalty)

    score = round(base + pr_points - penalty)
    score = max(0, min(100, score))

    parts = []
    for cat, count in sorted(category_counts.items()):
        if count > 0:
            w = _CATEGORY_WEIGHTS.get(cat, 0.5)
            parts.append(f"{cat}: {count} x {w}")
    explanation = (
        f"base {base} + PR points {pr_points:.1f} (capped at 50) - penalties {penalty}"
    )
    if parts:
        explanation += f" | Breakdown: {', '.join(parts)}"

    return score, explanation


def velocity_score_v2(
    merged_prs: list[tuple[str, GitHubMergedPR]],
    unique_repos_with_merges: int,
    drift_penalty: int,
) -> tuple[int, str]:
    """
    Compute velocity score based on PR throughput, complexity, and repo breadth.

    Returns (score, explanation_string).
    """
    base = 55
    points = 0.0

    for _repo, pr in merged_prs:
        points += 2.0

        # Complexity bonus based on net lines changed
        net = pr.additions + pr.deletions
        if 201 <= net <= 800:
            points += 0.5
        elif 801 <= net <= 2000:
            points += 1.0
        elif 2001 <= net <= 6000:
            points += 1.5
        elif net > 6000:
            points += 2.0

    # Repo breadth bonus
    points += min(unique_repos_with_merges, 8) * 1.0

    penalty = abs(drift_penalty)

    score = round(base + min(points, 45.0) - penalty)
    score = max(0, min(100, score))

    explanation = (
        f"base {base} + PR throughput/complexity ({len(merged_prs)} PRs)"
        f" + repo breadth ({min(unique_repos_with_merges, 8)} repos)"
        f" - drift penalty {abs(drift_penalty)}"
    )

    return score, explanation


def base_week_monday(date: dt.date) -> dt.date:
    # Monday is 0
    return date - dt.timedelta(days=date.weekday())


def month_name(date: dt.date) -> str:
    return date.strftime("%B").upper()


def deep_dive_filename(date: dt.date) -> str:
    return f"{month_name(date)}_{date.day}_{date.year}_DEEP_DIVE.md"


def sectionize_highlights(commits: Iterable[CommitEntry]) -> dict[str, list[str]]:
    # Lightweight classifier for "Major Components & Work Completed"
    buckets: dict[str, list[str]] = {
        "Runtime / Dispatch": [],
        "Models / Contracts": [],
        "Validation / CI Gates": [],
        "Idempotency / Time / Traceability": [],
        "Documentation / Planning": [],
        "Other": [],
    }
    for c in commits:
        s = c.subject.lower()
        item = f"{c.subject}"
        if any(
            k in s
            for k in [
                "dispatch",
                "dispatcher",
                "runtime",
                "kernel",
                "registry",
                "scheduler",
            ]
        ):
            buckets["Runtime / Dispatch"].append(item)
        elif any(k in s for k in ["model", "models", "contract", "schema", "envelope"]):
            buckets["Models / Contracts"].append(item)
        elif any(
            k in s
            for k in ["validator", "validation", "ci", "gate", "strict validation"]
        ):
            buckets["Validation / CI Gates"].append(item)
        elif any(
            k in s
            for k in [
                "idempot",
                "correlation",
                "causation",
                "trace",
                "time injection",
                "timeout",
            ]
        ):
            buckets["Idempotency / Time / Traceability"].append(item)
        elif any(
            k in s for k in ["docs", "document", "plan", "handoff", "adr", "readme"]
        ):
            buckets["Documentation / Planning"].append(item)
        else:
            buckets["Other"].append(item)

    # prune empty buckets
    return {k: v for k, v in buckets.items() if v}


@dataclass(frozen=True)
class ActiveWorktree:
    repo_name: str
    worktree_path: str
    branch: str
    head: str  # short commit hash or "(bare)"


def get_active_worktrees(repo: Path, repo_name: str) -> list[ActiveWorktree]:
    """
    Return non-main worktrees for a repo via `git worktree list --porcelain`.

    The main worktree (the bare clone or the primary checkout) is excluded
    since it already appears in the repo-day commit log.  Only feature-branch
    worktrees are returned — these represent in-progress work that may not
    have any commits today and would otherwise be invisible in the report.

    Returns an empty list on any error (allow_fail semantics).
    """
    raw = _run(["git", "worktree", "list", "--porcelain"], repo, allow_fail=True)
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


def is_primary_onex_repo(repo_name: str) -> bool:
    if repo_name.startswith("omnibase_core"):
        return True
    if repo_name.startswith("omnibase_infra"):
        return True
    if repo_name in {"omnibase_spi", "onex_change_control", "omninode_infra"}:
        return True
    return False


def repo_commit_sums(rd: RepoDay) -> tuple[int, int, int, int]:
    # commits, files, insertions, deletions
    files = ins = dele = 0
    for c in rd.commits:
        files += c.files
        ins += c.ins
        dele += c.dele
    return len(rd.commits), files, ins, dele


def unique_commit_entries(repo_days: list[RepoDay]) -> list[CommitEntry]:
    """
    Dedupe commit entries across parallel working copies.

    Key policy:
    - For omnibase_core* clones: dedupe by ('omnibase_core', full_hash)
    - For omnibase_infra* clones: dedupe by ('omnibase_infra', full_hash)
    - For all other repos: dedupe by (repo_name, full_hash)
    """
    seen: set[tuple[str, str]] = set()
    uniq: list[CommitEntry] = []
    for rd in sorted(repo_days, key=lambda x: x.name.lower()):
        fk = family_key(rd.name) or rd.name
        for c in rd.commits:
            k = (fk, c.full)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    _default_root = os.environ.get("OMNI_HOME", "/Volumes/PRO-G40/Code/omni_home")
    ap.add_argument(
        "--root",
        type=str,
        default=_default_root,
        help="Workspace root to scan (direct children only). Defaults to $OMNI_HOME.",
    )
    ap.add_argument(
        "--date", type=str, default=None, help="YYYY-MM-DD; defaults to today (local)."
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path. Defaults to <root>/docs/deep-dives/<MONTH>_<DAY>_<YEAR>_DEEP_DIVE.md",
    )
    ap.add_argument(
        "--json-scan-out",
        type=str,
        default=None,
        help="Optional path to write the repo scan JSON.",
    )
    ap.add_argument(
        "--include-dirty",
        action="store_true",
        default=False,
        help="Include repos that only have dirty working trees (no commits today). Default: False.",
    )
    args = ap.parse_args()

    date = _today_local() if not args.date else dt.date.fromisoformat(args.date)
    start_s, end_s = _day_window(date)

    root = Path(args.root).expanduser().resolve()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (root / "docs" / "deep-dives" / deep_dive_filename(date))
    )

    repos = find_git_repos_direct_children(root)

    # Filter to only include ONEX ecosystem repos
    _ONEX_REPO_PREFIXES = ("omni", "onex_")
    repos = [
        r
        for r in repos
        if any(r.name.lower().startswith(p) for p in _ONEX_REPO_PREFIXES)
    ]

    repo_days: list[RepoDay] = []
    all_active_worktrees: list[ActiveWorktree] = []
    scan_json: dict[str, object] = {
        "date": str(date),
        "start": start_s,
        "end": end_s,
        "repos_modified_today": [],
    }

    for repo in repos:
        name = repo.name
        commits = get_commit_entries(repo, start_s, end_s)
        merges = get_merge_entries(repo, start_s, end_s)
        dirty = get_dirty(repo)
        github_prs = get_github_merged_prs(repo, date)

        # Only include repos with commits today OR merged PRs (unless --include-dirty is set)
        if not commits and not github_prs:
            if not args.include_dirty or not dirty:
                continue

        branch = get_branch(repo)
        all_active_worktrees.extend(get_active_worktrees(repo, name))
        repo_days.append(
            RepoDay(
                name=name,
                path=repo,
                branch=branch,
                commits=commits,
                merges=merges,
                dirty=dirty,
                github_merged_prs=github_prs,
            )
        )
        scan_json["repos_modified_today"].append(
            {
                "repo": str(repo),
                "name": name,
                "commits_today": len(commits),
                "merged_prs_today": len(github_prs),
                "feature_prs_today": len(
                    [p for p in github_prs if not p.is_workflow_pr]
                ),
                "dirty": dirty,
            }
        )

    if args.json_scan_out:
        Path(args.json_scan_out).expanduser().resolve().write_text(
            json.dumps(scan_json, indent=2) + "\n"
        )

    # Aggregate family metrics (dedupe by full hash within family)
    families: dict[str, list[RepoDay]] = {
        "omnibase_core": [],
        "omnibase_infra": [],
        "omnimemory": [],
    }
    standalones: list[RepoDay] = []
    for rd in repo_days:
        fk = family_key(rd.name)
        if fk and fk in families:
            families[fk].append(rd)
        else:
            standalones.append(rd)

    def family_unique_commits(fam: list[RepoDay]) -> set[str]:
        s: set[str] = set()
        for rd in fam:
            for c in rd.commits:
                s.add(c.full)
        return s

    core_unique = len(family_unique_commits(families["omnibase_core"]))
    infra_unique = len(family_unique_commits(families["omnibase_infra"]))

    # Count PRs by category - DEDUPED by family + PR number
    seen_pr_keys: set[tuple[str, int]] = set()
    category_counts: dict[str, int] = {
        "capability": 0,
        "correctness": 0,
        "governance": 0,
        "observability": 0,
        "docs": 0,
        "churn": 0,
    }
    all_deduped_prs: list[tuple[str, GitHubMergedPR]] = []
    for rd in repo_days:
        fk = family_key(rd.name) or rd.name
        for pr in rd.github_merged_prs:
            if pr.is_workflow_pr:
                continue
            pr_key = (fk, pr.number)
            if pr_key not in seen_pr_keys:
                seen_pr_keys.add(pr_key)
                category_counts[pr.category] = category_counts.get(pr.category, 0) + 1
                all_deduped_prs.append((fk, pr))

    # Compute drift
    drift = compute_drift(repo_days, root, date)

    # Unique repos with merged PRs
    unique_repos_with_merges = len({repo for repo, _pr in all_deduped_prs})

    # V2 Scoring
    velocity, velocity_explanation = velocity_score_v2(
        all_deduped_prs, unique_repos_with_merges, drift.penalty
    )
    effectiveness, effectiveness_explanation = effectiveness_score_v2(
        category_counts, drift.penalty
    )

    # Begin document
    lines: list[str] = []
    lines.append(
        f"# {date.strftime('%B')} {date.day}, {date.year} - Deep Dive Analysis"
    )
    lines.append("")
    lines.append(
        f"**Date**: {date.strftime('%A')}, {date.strftime('%B')} {date.day}, {date.year}  "
    )
    monday = base_week_monday(date)
    lines.append(
        f"**Week**: Week of {monday.strftime('%B')} {monday.day}, {monday.year}  "
    )
    lines.append(f"**Day of Week**: {date.strftime('%A')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Velocity Score**: {velocity}/100  ")
    lines.append(f"**Effectiveness Score**: {effectiveness}/100")
    lines.append("")

    # Build dynamic assessment
    top_cats = sorted(
        ((cat, cnt) for cat, cnt in category_counts.items() if cnt > 0),
        key=lambda x: x[1],
        reverse=True,
    )
    if top_cats:
        top_names = [f"{cat} ({cnt})" for cat, cnt in top_cats[:3]]
        assessment = f"Top focus areas: {', '.join(top_names)}. Drift: {drift.level}."
    else:
        assessment = "No merged PRs detected for this period."
    lines.append(f"**Overall Assessment**: {assessment}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Merged PRs from GitHub (the "honest" metric)
    # Dedupe across working copies: use family key + PR number
    _CATEGORY_DISPLAY = {
        "capability": ("Capability (Net-New)", "🚀"),
        "correctness": ("Correctness / Hardening", "🔧"),
        "governance": ("Governance / Safety Rails", "🛡️"),
        "observability": ("Observability", "📊"),
        "docs": ("Documentation", "📝"),
        "churn": ("Churn / Investigation", "🔄"),
    }

    # Group PRs by category for display
    prs_by_category: dict[str, list[tuple[str, GitHubMergedPR]]] = {}
    workflow_prs: list[tuple[str, GitHubMergedPR]] = []
    seen_prs: set[tuple[str, int]] = set()

    for rd in repo_days:
        fk = family_key(rd.name) or rd.name
        for pr in rd.github_merged_prs:
            dedupe_key = (fk, pr.number)
            if dedupe_key in seen_prs:
                continue
            seen_prs.add(dedupe_key)
            display_name = fk if family_key(rd.name) else rd.name

            if pr.is_workflow_pr:
                workflow_prs.append((display_name, pr))
            else:
                prs_by_category.setdefault(pr.category, []).append((display_name, pr))

    has_any_prs = any(prs_by_category.values()) or workflow_prs
    if has_any_prs:
        lines.append("## Merged PRs (from GitHub)")
        lines.append("")
        lines.append(
            "*PRs categorized into 6 buckets to separate capability from correctness, governance, observability, docs, and churn.*"
        )
        lines.append("")

        # Render each non-empty category
        for cat_key in [
            "capability",
            "correctness",
            "governance",
            "observability",
            "docs",
            "churn",
        ]:
            cat_prs = prs_by_category.get(cat_key, [])
            if not cat_prs:
                continue
            display_label, emoji = _CATEGORY_DISPLAY[cat_key]
            lines.append(f"### {emoji} {display_label}")
            lines.append("")
            lines.append("| Repo | PR | Title | Merged |")
            lines.append("|------|-----|-------|--------|")
            for repo_name, pr in sorted(cat_prs, key=lambda x: x[1].merged_at):
                merge_time = (
                    pr.merged_at[11:16] if len(pr.merged_at) > 16 else pr.merged_at
                )
                lines.append(
                    f"| {repo_name} | #{pr.number} | {pr.title} | {merge_time} EST |"
                )
            lines.append("")
            lines.append(f"**{display_label} PRs**: {len(cat_prs)}")
            lines.append("")

        # Workflow noise (filtered)
        if workflow_prs:
            lines.append("### Workflow/Automation (excluded from scoring)")
            lines.append("")
            lines.append(f"*{len(workflow_prs)} auto-generated workflow PRs merged*")
            lines.append("")

        # Summary
        lines.append("### Summary")
        lines.append("")
        for cat_key in [
            "capability",
            "correctness",
            "governance",
            "observability",
            "docs",
            "churn",
        ]:
            display_label, _emoji = _CATEGORY_DISPLAY[cat_key]
            count = category_counts.get(cat_key, 0)
            lines.append(f"- **{display_label}**: {count} PRs")
        lines.append(f"- **Workflow noise**: {len(workflow_prs)} PRs (excluded)")
        lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Repository Activity Overview (Repos with commits today)")
    lines.append("")
    # Core rollup (only working copies with commits)
    core_rds_with_commits = [rd for rd in families["omnibase_core"] if rd.commits]
    core_rds_with_commits = sorted(core_rds_with_commits, key=lambda x: x.name.lower())
    if core_rds_with_commits:
        lines.append("### omnibase_core (across working copies)")
        lines.append(
            f"- **Unique commits (deduped across working copies)**: {core_unique}"
        )
        lines.append("- **Working-copy activity**:")
        for rd in core_rds_with_commits:
            c, f, ins, dele = repo_commit_sums(rd)
            lines.append(f"  - `{rd.name}`: {c} commits | {f} files | +{ins} / -{dele}")
        lines.append("")

    # Infra rollup (only working copies with commits)
    infra_rds_with_commits = [rd for rd in families["omnibase_infra"] if rd.commits]
    infra_rds_with_commits = sorted(
        infra_rds_with_commits, key=lambda x: x.name.lower()
    )
    if infra_rds_with_commits:
        lines.append("### omnibase_infra (across working copies)")
        lines.append(
            f"- **Unique commits (deduped across working copies)**: {infra_unique}"
        )
        lines.append("- **Working-copy activity**:")
        for rd in infra_rds_with_commits:
            c, f, ins, dele = repo_commit_sums(rd)
            lines.append(f"  - `{rd.name}`: {c} commits | {f} files | +{ins} / -{dele}")
        lines.append("")

    # Other primary ONEX repos (only those with commits)
    primary_other = [
        rd
        for rd in repo_days
        if rd.name in {"omnibase_spi", "onex_change_control", "omninode_infra"}
        and rd.commits
    ]
    if primary_other:
        lines.append("### Other ONEX ecosystem repos")
        for rd in sorted(primary_other, key=lambda x: x.name.lower()):
            c, f, ins, dele = repo_commit_sums(rd)
            lines.append(
                f"- **{rd.name}**: {c} commits | {f} files | +{ins} / -{dele} (branch: `{rd.branch}`)"
            )
        lines.append("")

    # Other repos (only those with commits)
    other_repos = [
        rd for rd in repo_days if not is_primary_onex_repo(rd.name) and rd.commits
    ]
    if other_repos:
        lines.append("### Other repos with commits today")
        for rd in sorted(other_repos, key=lambda x: x.name.lower()):
            c, f, ins, dele = repo_commit_sums(rd)
            lines.append(
                f"- **{rd.name}**: {c} commits | {f} files | +{ins} / -{dele} (branch: `{rd.branch}`)"
            )
        lines.append("")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Active worktrees (in-progress branches not yet merged)
    if all_active_worktrees:
        lines.append("## Active Worktrees (In-Progress Branches)")
        lines.append("")
        lines.append(
            "*Worktrees represent active development branches that may have no commits today.*"
        )
        lines.append("")
        lines.append("| Repo | Branch | Head | Path |")
        lines.append("|------|--------|------|------|")
        for wt in sorted(all_active_worktrees, key=lambda w: (w.repo_name, w.branch)):
            lines.append(
                f"| {wt.repo_name} | `{wt.branch}` | `{wt.head}` | `{wt.worktree_path}` |"
            )
        lines.append("")
        lines.append(f"**Total active worktrees**: {len(all_active_worktrees)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Major components: use ONEX-ish subset if present
    lines.append("## Major Components & Work Completed")
    lines.append("")
    primary_repo_days = [rd for rd in repo_days if is_primary_onex_repo(rd.name)]
    uniq_primary_commits = unique_commit_entries(primary_repo_days)
    buckets = sectionize_highlights(uniq_primary_commits)
    for title, items in buckets.items():
        lines.append(f"### {title}")
        for it in items[:12]:
            lines.append(f"- {it}")
        if len(items) > 12:
            lines.append(f"- ... ({len(items) - 12} more)")
        lines.append("")
    lines.append("---")
    lines.append("")

    # PR Inventory: inferred from subjects (best-effort; local-only)
    lines.append("## PR Inventory (Today)")
    lines.append("")
    for rd in sorted(primary_repo_days, key=lambda x: x.name.lower()):
        prs: set[int] = set()
        for c in rd.commits:
            prs.update(extract_pr_numbers(c.subject))
        for m in rd.merges:
            prs.update(extract_pr_numbers(m.subject))
        if not prs:
            continue
        lines.append(f"### {rd.name}")
        lines.append(
            f"- PR references in commits: {', '.join(f'#{p}' for p in sorted(prs))}"
        )
        lines.append("")
    # Local dirty-only repos (no commits) — only shown if --include-dirty was used
    if args.include_dirty:
        dirty_only = [rd for rd in repo_days if rd.dirty and not rd.commits]
        if dirty_only:
            lines.append("### Stale repos (dirty working tree only; no commits today)")
            for rd in sorted(dirty_only, key=lambda x: x.name.lower()):
                lines.append(
                    f"- **{rd.name}**: {len(rd.dirty)} uncommitted changes (branch: `{rd.branch}`)"
                )
            lines.append("")
    lines.append("---")
    lines.append("")

    # Ticket Summary
    all_tickets = collect_all_ticket_ids(repo_days)
    if all_tickets:
        lines.append("## Ticket Summary")
        lines.append("")
        lines.append(
            f"**{len(all_tickets)} unique tickets** referenced in today's commits:"
        )
        lines.append("")
        # Group into rows of 8 for readability
        for i in range(0, len(all_tickets), 8):
            chunk = all_tickets[i : i + 8]
            lines.append("  " + ", ".join(chunk))
        lines.append("")
        lines.append("---")
        lines.append("")

    # Metrics
    lines.append("## Metrics & Statistics")
    lines.append("")
    lines.append("### Commit Activity")
    lines.append(f"- **Core (all working copies)**: {core_unique} unique commits")
    lines.append(f"- **Infra (all working copies)**: {infra_unique} unique commits")
    lines.append(
        f"- **Other repos**: {sum(len(rd.commits) for rd in standalones)} commits (non-deduped)"
    )
    lines.append("")

    def sum_stats(rds: list[RepoDay]) -> tuple[int, int, int]:
        files = ins = dele = 0
        for rd in rds:
            for c in rd.commits:
                files += c.files
                ins += c.ins
                dele += c.dele
        return files, ins, dele

    core_files, core_ins, core_del = sum_stats(families["omnibase_core"])
    infra_files, infra_ins, infra_del = sum_stats(families["omnibase_infra"])

    lines.append("### File & Line Changes (from today's commits; summed)")
    lines.append(f"- **Core**: {core_files} files | +{core_ins} / -{core_del}")
    lines.append(f"- **Infra**: {infra_files} files | +{infra_ins} / -{infra_del}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Risks (only consider repos with commits today)
    repos_with_commits = [rd for rd in repo_days if rd.commits]
    lines.append("## Challenges / Risks Observed")
    lines.append("")
    added_risk = False
    if any(any(".env" in d for d in rd.dirty) for rd in repos_with_commits):
        lines.append(
            "- **Environment/config churn**: `.env`/docker config touched in repos with today's commits; easy source of local divergence."
        )
        added_risk = True
    if not added_risk:
        lines.append(
            "- No major risks detected from commit metadata alone; integration/flake risk remains once real infra is exercised."
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Drift Telemetry
    lines.append("## Drift Telemetry")
    lines.append("")
    drift_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(drift.level, "⚪")
    lines.append(
        f"**Drift Score**: {drift_emoji} {drift.level.upper()} (penalty: {drift.penalty})"
    )
    lines.append("")
    lines.append("| Signal | Count | Risk Level |")
    lines.append("|--------|-------|------------|")
    lines.append(
        f"| Dirty files on main/master | {drift.main_dirty} | {'⚠️ HIGH' if drift.main_dirty > 0 else '✅'} |"
    )
    lines.append(
        f"| Stale feature branches (>72h) | {drift.stale_branches} | {'⚠️ MEDIUM' if drift.stale_branches > 0 else '✅'} |"
    )
    lines.append(
        f"| Diverged from origin/main (>48h) | {drift.diverged_branches} | {'ℹ️ INFO' if drift.diverged_branches > 0 else '✅'} |"  # noqa: RUF001
    )
    lines.append(
        f"| Active parallel worktrees | {drift.active_worktrees} | ✅ (normal) |"
    )
    lines.append("")
    if drift.risks:
        lines.append("**Action items**:")
        for repo_name, reason in drift.risks:
            lines.append(f"- `{repo_name}`: {reason}")
        lines.append("")
    else:
        lines.append("No drift risks detected.")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Demo Readiness (stub -- requires golden path runner data)
    lines.append("## Demo Readiness")
    lines.append("")
    lines.append(
        "*Data sources not yet available. Requires golden path runner integration.*"
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append("| Golden path runs today | — |")
    lines.append("| Success rate (last 10) | — |")
    lines.append("| Manual steps remaining | — |")
    lines.append(
        f"| Critical path tickets remaining | {category_counts.get('capability', 0)} (approx) |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Velocity & Effectiveness Scoring")
    lines.append("")
    lines.append(f"**Velocity Score: {velocity}/100**")
    lines.append(f"- {velocity_explanation}")
    lines.append("")
    lines.append(f"**Effectiveness Score: {effectiveness}/100**")
    lines.append(f"- {effectiveness_explanation}")
    lines.append("")
    lines.append("### PR Category Breakdown")
    lines.append("")
    lines.append("| Category | Count | Weight | Points |")
    lines.append("|----------|-------|--------|--------|")
    for cat in [
        "capability",
        "correctness",
        "governance",
        "observability",
        "docs",
        "churn",
    ]:
        count = category_counts.get(cat, 0)
        weight = _CATEGORY_WEIGHTS.get(cat, 0.5)
        points = count * weight
        lines.append(f"| {cat} | {count} | {weight} | {points:.1f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Appendix A — Complete Commit Logs (Every Commit Today)")
    lines.append("")
    for rd in sorted(repo_days, key=lambda x: x.name.lower()):
        if not rd.commits:
            continue
        lines.append(f"### {rd.name} (full day log capture)")
        lines.append("```")
        for c in rd.commits:
            lines.append(f"{c.short}|{c.ai}|{c.author}|{c.subject}")
        lines.append("```")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

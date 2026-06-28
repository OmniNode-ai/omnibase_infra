# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared data models for the ONEX deep-dive report (OMN-13725).

These are pure Python dataclasses — no I/O, no omnibase imports — so they
can be imported by scripts, nodes, and tests without pulling in any
infrastructure dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CommitEntry:
    """A single git commit with stat information."""

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
    """A merge commit (subset of CommitEntry, no stat lines)."""

    full: str
    short: str
    ai: str
    author: str
    subject: str


@dataclass(frozen=True)
class GitHubMergedPR:
    """A PR merged via the GitHub API (from ``gh pr list``)."""

    number: int
    title: str
    merged_at: str
    is_workflow_pr: bool
    category: (
        str  # capability | correctness | governance | observability | docs | churn
    )
    is_exempt: bool = False  # dep bump or release; exempt from ticket-linkage check
    additions: int = 0
    deletions: int = 0


@dataclass(frozen=True)
class RepoDay:
    """A git repository with activity for a single calendar day."""

    name: str
    path: Path
    branch: str
    commits: list[CommitEntry] = field(default_factory=list)
    merges: list[MergeEntry] = field(default_factory=list)
    dirty: list[str] = field(default_factory=list)
    github_merged_prs: list[GitHubMergedPR] = field(default_factory=list)


@dataclass(frozen=True)
class DriftReport:
    """Risk signals derived from repository state for the report period."""

    level: str  # green | yellow | red
    main_dirty: int  # clones on main with uncommitted changes
    stale_branches: int  # feature branches with last commit >72h ago
    diverged_branches: int  # feature branches diverged from origin/main >48h
    risks: list[tuple[str, str]]  # (repo_name, reason)
    penalty: int  # 0, -2, or -5
    # Informational (not scored)
    active_worktrees: int
    unlinked_pr_count: int = 0
    total_pr_count: int = 0


@dataclass(frozen=True)
class ActiveWorktree:
    """A non-main git worktree representing in-progress work."""

    repo_name: str
    worktree_path: str
    branch: str
    head: str  # short commit hash or "(bare)"


__all__: list[str] = [
    "ActiveWorktree",
    "CommitEntry",
    "DriftReport",
    "GitHubMergedPR",
    "MergeEntry",
    "RepoDay",
]

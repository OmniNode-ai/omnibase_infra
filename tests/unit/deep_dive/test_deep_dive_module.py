# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the shared ``omnibase_infra.deep_dive`` module (OMN-13725).

These lock the public surface that both ``generate_deep_dive.py`` and
``omnimarket.nodes.node_deep_dive_report_effect`` import from.  The
``compute_drift`` cases use ``main``-branch repo-days so no git subprocess is
invoked (the feature-branch path that shells out is not exercised here).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from omnibase_infra.deep_dive import (
    CommitEntry,
    GitHubMergedPR,
    RepoDay,
    classify_pr,
    compute_drift,
    family_key,
    unique_commit_entries,
)

pytestmark = pytest.mark.unit


def _commit(full: str, subject: str = "feat: x") -> CommitEntry:
    return CommitEntry(
        full=full,
        short=full[:8],
        ai="2026-06-28 10:00:00 +0000",
        author="agent",
        subject=subject,
        files=1,
        ins=1,
        dele=0,
    )


def _repo_day(
    name: str,
    *,
    branch: str = "main",
    commits: list[CommitEntry] | None = None,
    dirty: list[str] | None = None,
    prs: list[GitHubMergedPR] | None = None,
) -> RepoDay:
    return RepoDay(
        name=name,
        path=Path("/nonexistent") / name,
        branch=branch,
        commits=commits or [],
        merges=[],
        dirty=dirty or [],
        github_merged_prs=prs or [],
    )


class TestFamilyDedup:
    def test_family_key_strips_trailing_digits(self) -> None:
        assert family_key("omnibase_core2") == "omnibase_core"
        assert family_key("omnidash4") == "omnidash"
        assert family_key("omnibase_core") == "omnibase_core"
        assert family_key("unrelated") is None

    def test_unique_commit_entries_dedupes_across_clones(self) -> None:
        shared = _commit("a" * 40)
        rd1 = _repo_day("omnibase_core", commits=[shared])
        rd2 = _repo_day("omnibase_core2", commits=[shared, _commit("b" * 40)])
        uniq = unique_commit_entries([rd1, rd2])
        # The shared commit appears once despite being in both clones.
        assert sum(1 for c in uniq if c.full == "a" * 40) == 1
        assert len(uniq) == 2


class TestComputeDriftPurePath:
    def test_clean_main_is_green(self) -> None:
        rd = _repo_day("omnibase_infra", commits=[_commit("c" * 40)])
        report = compute_drift([rd], Path("/nonexistent"), dt.date(2026, 6, 28))
        assert report.level == "green"
        assert report.penalty == 0
        assert report.main_dirty == 0

    def test_dirty_main_raises_drift(self) -> None:
        rd = _repo_day(
            "omnibase_infra",
            branch="main",
            commits=[_commit("d" * 40)],
            dirty=["M file_a", "M file_b"],
        )
        report = compute_drift([rd], Path("/nonexistent"), dt.date(2026, 6, 28))
        assert report.main_dirty == 1
        # main_dirty * 3 == 3 -> yellow band
        assert report.level == "yellow"
        assert report.penalty == -2

    def test_unlinked_pr_counted(self) -> None:
        linked = GitHubMergedPR(
            number=1,
            title="feat(OMN-1): wire it",
            merged_at="10:00",
            is_workflow_pr=False,
            category=classify_pr("feat(OMN-1): wire it"),
            is_exempt=False,
        )
        unlinked = GitHubMergedPR(
            number=2,
            title="feat: no ticket here",
            merged_at="11:00",
            is_workflow_pr=False,
            category=classify_pr("feat: no ticket here"),
            is_exempt=False,
        )
        rd = _repo_day("omnibase_infra", prs=[linked, unlinked])
        report = compute_drift([rd], Path("/nonexistent"), dt.date(2026, 6, 28))
        assert report.total_pr_count == 2
        assert report.unlinked_pr_count == 1

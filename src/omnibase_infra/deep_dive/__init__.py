# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared deep-dive data models and pure-processing functions (OMN-13725).

This package extracts the portable, I/O-free parts of
``omnibase_infra/scripts/generate_deep_dive.py`` into an importable
module so that:

- ``scripts/generate_deep_dive.py`` can import from here (one source of
  truth, no copy-paste drift), and
- ``omnimarket.nodes.node_deep_dive_report_effect`` can import the same
  models and scoring functions via the injected ``ProtocolReportDataSource``
  adapter, with all git/``gh`` I/O contained at the EFFECT boundary.

Public surface:
    - Data models: ``CommitEntry``, ``MergeEntry``, ``GitHubMergedPR``,
      ``RepoDay``, ``DriftReport``, ``ActiveWorktree``
    - Pure functions: ``classify_pr``, ``is_workflow_pr``, ``is_exempt_pr``,
      ``extract_pr_numbers``, ``extract_ticket_ids``,
      ``collect_all_ticket_ids``, ``family_key``,
      ``effectiveness_score_v2``, ``velocity_score_v2``,
      ``sectionize_highlights``, ``unique_commit_entries``,
      ``repo_commit_sums``, ``is_primary_onex_repo``,
      ``base_week_monday``, ``month_name``, ``deep_dive_filename``,
      ``parse_runtime_state_timestamp``, ``validate_deep_dive_ingest``,
      ``CATEGORY_WEIGHTS``
"""

from omnibase_infra.deep_dive.models import (
    ActiveWorktree,
    CommitEntry,
    DriftReport,
    GitHubMergedPR,
    MergeEntry,
    RepoDay,
)
from omnibase_infra.deep_dive.processing import (
    CATEGORY_WEIGHTS,
    base_week_monday,
    classify_pr,
    collect_all_ticket_ids,
    deep_dive_filename,
    effectiveness_score_v2,
    extract_pr_numbers,
    extract_ticket_ids,
    family_key,
    is_exempt_pr,
    is_primary_onex_repo,
    is_workflow_pr,
    month_name,
    parse_runtime_state_timestamp,
    repo_commit_sums,
    sectionize_highlights,
    unique_commit_entries,
    validate_deep_dive_ingest,
    velocity_score_v2,
)
from omnibase_infra.deep_dive.scan import (
    compute_drift,
    find_git_repos_direct_children,
    get_active_worktrees,
    get_branch,
    get_commit_entries,
    get_dirty,
    get_github_merged_prs,
    get_merge_entries,
    run_git,
)

__all__: list[str] = [
    # Data models
    "ActiveWorktree",
    "CommitEntry",
    "DriftReport",
    "GitHubMergedPR",
    "MergeEntry",
    "RepoDay",
    # Pure functions
    "CATEGORY_WEIGHTS",
    "base_week_monday",
    "classify_pr",
    "collect_all_ticket_ids",
    "deep_dive_filename",
    "effectiveness_score_v2",
    "extract_pr_numbers",
    "extract_ticket_ids",
    "family_key",
    "is_exempt_pr",
    "is_primary_onex_repo",
    "is_workflow_pr",
    "month_name",
    "parse_runtime_state_timestamp",
    "repo_commit_sums",
    "sectionize_highlights",
    "unique_commit_entries",
    "validate_deep_dive_ingest",
    "velocity_score_v2",
    # Git/gh scan helpers (subprocess I/O boundary)
    "compute_drift",
    "find_git_repos_direct_children",
    "get_active_worktrees",
    "get_branch",
    "get_commit_entries",
    "get_dirty",
    "get_github_merged_prs",
    "get_merge_entries",
    "run_git",
]

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
  uv run python scripts/generate_deep_dive.py
  uv run python scripts/generate_deep_dive.py --date 2025-12-20
  uv run python scripts/generate_deep_dive.py --root $OMNI_HOME --out /tmp/DECEMBER_20_2025_DEEP_DIVE.md
  uv run python scripts/generate_deep_dive.py --include-dirty

Pure functions (PR categorisation, scoring, dedup, etc.) live in
``omnibase_infra.deep_dive`` and are importable by other packages (e.g.
``omnimarket.nodes.node_deep_dive_report_effect``).  This script owns
all git/``gh`` I/O and the CLI entry-point only; it imports from the
shared module rather than duplicating logic (OMN-13725).
"""

from __future__ import annotations

# Allow standalone execution without `uv run` (adds src/ to path when the
# package is not yet installed in the active env).
import sys
from pathlib import Path as _Path

_REPO_SRC = _Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path and _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

import argparse
import datetime as dt
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared pure models and processing functions (one source of truth)
# ---------------------------------------------------------------------------
from omnibase_infra.deep_dive import (
    CATEGORY_WEIGHTS,
    ActiveWorktree,
    GitHubMergedPR,
    RepoDay,
    base_week_monday,
    collect_all_ticket_ids,
    compute_drift,
    deep_dive_filename,
    effectiveness_score_v2,
    extract_pr_numbers,
    family_key,
    find_git_repos_direct_children,
    get_active_worktrees,
    get_branch,
    get_commit_entries,
    get_dirty,
    get_github_merged_prs,
    get_merge_entries,
    is_primary_onex_repo,
    repo_commit_sums,
    sectionize_highlights,
    unique_commit_entries,
    validate_deep_dive_ingest,
    velocity_score_v2,
)


def _today_local() -> dt.date:
    return dt.datetime.now().astimezone().date()


def _day_window(date: dt.date) -> tuple[str, str]:
    # Local time window, consistent with most deep-dive narratives.
    start = dt.datetime.combine(date, dt.time(0, 0, 0))
    end = dt.datetime.combine(date, dt.time(23, 59, 59))
    return (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))


def _now_utc_iso_minutes() -> str:
    """Return current UTC time as YYYY-MM-DDTHH:MMZ (minute precision)."""
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    _default_root = os.environ.get("OMNI_HOME", ".")
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
    ap.add_argument(
        "--validate-ingest",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Validate an existing deep-dive for corpus ingestion. "
            "Exits 0 if the '## Runtime State — as of' section is not stale "
            "relative to --handoff-time, non-zero otherwise."
        ),
    )
    ap.add_argument(
        "--handoff-time",
        type=str,
        default=None,
        metavar="ISO-TIMESTAMP",
        help=(
            "UTC probe time of the corresponding final handoff, "
            "e.g. '2026-06-28T14:30Z'. Required when --validate-ingest is used."
        ),
    )
    args = ap.parse_args()

    # Validation mode: check an existing deep-dive for staleness and exit.
    if args.validate_ingest is not None:
        if not args.handoff_time:
            print(
                "error: --handoff-time is required with --validate-ingest", flush=True
            )
            return 2
        try:
            cutoff = dt.datetime.fromisoformat(args.handoff_time.replace("Z", "+00:00"))
        except ValueError:
            print(
                f"error: --handoff-time '{args.handoff_time}' is not a valid ISO-8601 timestamp",
                flush=True,
            )
            return 2
        ok, msg = validate_deep_dive_ingest(
            Path(args.validate_ingest).expanduser().resolve(), cutoff
        )
        print(msg, flush=True)
        return 0 if ok else 1

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
    # Unlinked PR telemetry (OMN-6922)
    if drift.total_pr_count > 0:
        unlinked_pct = drift.unlinked_pr_count / drift.total_pr_count * 100
        unlinked_status = "⚠️ HIGH" if unlinked_pct > 5 else "✅"
        lines.append(
            f"| Unlinked PRs (no OMN-XXXX) | {drift.unlinked_pr_count}/{drift.total_pr_count} ({unlinked_pct:.0f}%) | {unlinked_status} |"
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
        weight = CATEGORY_WEIGHTS.get(cat, 0.5)
        points = count * weight
        lines.append(f"| {cat} | {count} | {weight} | {points:.1f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Typed sections: Durable Findings vs Runtime State (OMN-13043 / C-3)
    #
    # Durable Findings: structural observations from commit and PR history.
    # These remain valid until the referenced code is changed.
    #
    # Runtime State: ephemeral observations captured at generation time.
    # The '## Runtime State — as of' header carries a machine-readable
    # UTC timestamp used by --validate-ingest to gate corpus ingestion.
    # -----------------------------------------------------------------------
    lines.append("## Durable Findings")
    lines.append("")
    lines.append(
        "*Structural findings derived from commit and PR history in this report.*"
    )
    lines.append("*These remain valid until the referenced code is changed.*")
    lines.append("")
    lines.append(
        "<!-- Operator: promote session retrospective findings here. "
        "Each entry must cite a commit SHA or PR number as evidence. -->"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    _state_ts = _now_utc_iso_minutes()
    lines.append(f"## Runtime State — as of {_state_ts}")
    lines.append("")
    lines.append("*Ephemeral runtime observations captured at generation time.*")
    lines.append("*This section has a bounded lifetime — it becomes stale once the*")
    lines.append("*runtime state changes.  Use*")
    lines.append(
        "*`generate_deep_dive.py --validate-ingest <path> --handoff-time <ISO>`*"
    )
    lines.append("*to check freshness before corpus ingestion.*")
    lines.append("")
    drift_emoji2 = {"green": "green", "yellow": "yellow", "red": "red"}.get(
        drift.level, drift.level
    )
    lines.append(f"- **Drift level at generation**: {drift_emoji2}")
    lines.append(f"- **Main-dirty repos**: {drift.main_dirty}")
    lines.append(f"- **Stale feature branches (>72h)**: {drift.stale_branches}")
    lines.append(f"- **Active worktrees**: {drift.active_worktrees}")
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

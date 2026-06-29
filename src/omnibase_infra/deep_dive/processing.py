# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure deep-dive processing functions (OMN-13725).

All functions in this module are I/O-free and deterministic given the same
input data.  They operate on the data models defined in
``omnibase_infra.deep_dive.models`` and return only plain Python values.

No subprocess calls, no file I/O, no network access.  Any data that
requires git or ``gh`` must be fetched by the caller (the script entry
point or the EFFECT handler's injected adapter) and passed in.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from pathlib import Path

from omnibase_infra.deep_dive.models import (
    CommitEntry,
    DriftReport,
    GitHubMergedPR,
    RepoDay,
)

# ---------------------------------------------------------------------------
# Compiled patterns (module-level constants)
# ---------------------------------------------------------------------------

PR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(#(?P<num>\d+)\)"),
    re.compile(r"\bPR\s*#(?P<num>\d+)\b", re.IGNORECASE),
    re.compile(r"\b#(?P<num>\d+)\b"),
    re.compile(r"Merge pull request #(?P<num>\d+)", re.IGNORECASE),
]

TICKET_RE: re.Pattern[str] = re.compile(r"\b(OMN-\d+)\b")

WORKFLOW_PR_PATTERNS: list[str] = [
    "Add Claude Code GitHub Workflow",
    "Update Claude Code Review workflow",
    "Update Claude PR Assistant workflow",
]

# Exempt PR title prefixes (dep bumps / releases).
# SYNC: exempt prefixes must match the lists in:
#   - onex_change_control/.github/workflows/pr-title-check-reusable.yml
#   - omniclaude/src/omniclaude/nodes/node_git_effect/models/model_git_request.py
_EXEMPT_PREFIXES: tuple[str, ...] = (
    "chore(deps",
    "build(deps",
    "bump ",
    "chore: release",
    "chore(release)",
    "release:",
)

CATEGORY_WEIGHTS: dict[str, float] = {
    "capability": 2.0,
    "correctness": 1.5,
    "governance": 1.5,
    "observability": 1.2,
    "docs": 0.7,
    "churn": 0.2,
}

_FAMILY_RE: re.Pattern[str] = re.compile(r"^(omni\w+?)(\d+)$")

# Matches the typed Runtime State section header emitted by the generator.
# Group 1 captures the ISO-8601 UTC timestamp (minute precision).
# Example: ## Runtime State — as of 2026-06-28T14:30Z
_RUNTIME_STATE_RE: re.Pattern[str] = re.compile(
    r"^## Runtime State — as of (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# PR categorization helpers
# ---------------------------------------------------------------------------


def is_workflow_pr(title: str) -> bool:
    """Return True if the PR title matches a known auto-generated workflow pattern."""
    return any(pattern.lower() in title.lower() for pattern in WORKFLOW_PR_PATTERNS)


def is_exempt_pr(title: str) -> bool:
    """Return True if the PR is a dep bump or release (exempt from ticket linkage)."""
    return title.lower().startswith(_EXEMPT_PREFIXES)


def classify_pr(title: str) -> str:
    """Classify a PR into one of six deterministic categories.

    Categories: capability, correctness, governance, observability, docs, churn
    Override rules are checked first, then conventional-commit prefixes.
    """
    t = title.lower()

    # Override rules (take precedence over prefix classification)
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

    # Conventional-commit prefix classification
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

    return "correctness"


# ---------------------------------------------------------------------------
# Ticket / PR extraction
# ---------------------------------------------------------------------------


def extract_pr_numbers(subject: str) -> list[int]:
    """Extract PR numbers referenced in a git commit subject line."""
    prs: set[int] = set()
    for pat in PR_PATTERNS:
        for m in pat.finditer(subject):
            try:
                prs.add(int(m.group("num")))
            except Exception:  # noqa: BLE001
                continue
    return sorted(prs)


def extract_ticket_ids(subject: str) -> list[str]:
    """Extract OMN ticket identifiers (e.g. OMN-123) from a commit subject."""
    return sorted(set(TICKET_RE.findall(subject)))


def collect_all_ticket_ids(repo_days: list[RepoDay]) -> list[str]:
    """Collect all unique OMN ticket IDs (e.g. OMN-123) from a list of RepoDay objects."""
    ids: set[str] = set()
    for rd in repo_days:
        for c in rd.commits:
            ids.update(extract_ticket_ids(c.subject))
        for m in rd.merges:
            ids.update(extract_ticket_ids(m.subject))
    return sorted(ids, key=lambda x: int(x.split("-")[1]))


# ---------------------------------------------------------------------------
# Family deduplication helpers
# ---------------------------------------------------------------------------


def family_key(repo_name: str) -> str | None:
    """Map a repo clone to its canonical family name for deduplication.

    Strips trailing digits: ``omnibase_core2`` → ``omnibase_core``.
    Returns ``None`` for repos without trailing digits (already canonical).
    """
    m = _FAMILY_RE.match(repo_name)
    if m:
        return m.group(1)
    if repo_name.startswith("omni"):
        return repo_name
    return None


def unique_commit_entries(repo_days: list[RepoDay]) -> list[CommitEntry]:
    """Deduplicate CommitEntry objects across parallel working copies.

    Policy:
    - For omnibase_core* clones: deduplicate by (``omnibase_core``, full_hash)
    - For omnibase_infra* clones: deduplicate by (``omnibase_infra``, full_hash)
    - All others: deduplicate by (repo_name, full_hash)
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


def repo_commit_sums(rd: RepoDay) -> tuple[int, int, int, int]:
    """Return (commit_count, files_changed, insertions, deletions) for a RepoDay."""
    files = ins = dele = 0
    for c in rd.commits:
        files += c.files
        ins += c.ins
        dele += c.dele
    return len(rd.commits), files, ins, dele


def is_primary_onex_repo(repo_name: str) -> bool:
    """Return True for repos considered primary ONEX ecosystem repos."""
    if repo_name.startswith("omnibase_core"):
        return True
    if repo_name.startswith("omnibase_infra"):
        return True
    if repo_name in {"omnibase_spi", "onex_change_control", "omninode_infra"}:
        return True
    return False


# ---------------------------------------------------------------------------
# V2 Scoring
# ---------------------------------------------------------------------------


def effectiveness_score_v2(
    category_counts: dict[str, int],
    drift_penalty: int,
) -> tuple[int, str]:
    """Compute a 0-100 effectiveness score based on weighted PR categories.

    Returns ``(score, explanation_string)``.
    """
    base = 60
    pr_points = 0.0
    total_prs = sum(category_counts.values())

    for cat, count in category_counts.items():
        pr_points += CATEGORY_WEIGHTS.get(cat, 0.5) * count

    pr_points = min(pr_points, 50.0)

    penalty = 0
    churn_count = category_counts.get("churn", 0)
    churn_ratio = churn_count / max(total_prs, 1)
    if churn_ratio > 0.20:
        penalty += 3

    penalty += abs(drift_penalty)

    score = round(base + pr_points - penalty)
    score = max(0, min(100, score))

    parts = [
        f"{cat}: {count} x {CATEGORY_WEIGHTS.get(cat, 0.5)}"
        for cat, count in sorted(category_counts.items())
        if count > 0
    ]
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
    """Compute a 0-100 velocity score based on PR throughput and complexity.

    Returns ``(score, explanation_string)``.
    """
    base = 55
    points = 0.0

    for _repo, pr in merged_prs:
        points += 2.0
        net = pr.additions + pr.deletions
        if 201 <= net <= 800:
            points += 0.5
        elif 801 <= net <= 2000:
            points += 1.0
        elif 2001 <= net <= 6000:
            points += 1.5
        elif net > 6000:
            points += 2.0

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


# ---------------------------------------------------------------------------
# Date / filename helpers
# ---------------------------------------------------------------------------


def base_week_monday(date: dt.date) -> dt.date:
    """Return the Monday of the week containing *date*."""
    return date - dt.timedelta(days=date.weekday())


def month_name(date: dt.date) -> str:
    """Return the uppercase month name for a date, e.g. ``JUNE``."""
    return date.strftime("%B").upper()


def deep_dive_filename(date: dt.date) -> str:
    """Return the canonical deep-dive Markdown filename for a date."""
    return f"{month_name(date)}_{date.day}_{date.year}_DEEP_DIVE.md"


# ---------------------------------------------------------------------------
# Commit categorisation
# ---------------------------------------------------------------------------


def sectionize_highlights(commits: Iterable[CommitEntry]) -> dict[str, list[str]]:
    """Classify commit subjects into thematic buckets for the report."""
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
        item = c.subject
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
    return {k: v for k, v in buckets.items() if v}


# ---------------------------------------------------------------------------
# Runtime State freshness validation
# ---------------------------------------------------------------------------


def parse_runtime_state_timestamp(text: str) -> dt.datetime | None:
    """Extract the UTC datetime from a '## Runtime State — as of' header.

    Returns a UTC-aware ``datetime`` or ``None`` when the section is absent or
    the timestamp is malformed.
    """
    m = _RUNTIME_STATE_RE.search(text)
    if not m:
        return None
    try:
        return dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_deep_dive_ingest(
    deep_dive_path: Path,
    handoff_cutoff: dt.datetime,
) -> tuple[bool, str]:
    """Check that a deep-dive's Runtime State section is not stale.

    A deep-dive is safe to ingest when its ``## Runtime State — as of``
    timestamp is equal to or newer than *handoff_cutoff*.  A missing section
    is always refused.

    Returns ``(ok, message)``.
    """
    try:
        text = deep_dive_path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"cannot read {deep_dive_path}: {e}"

    ts = parse_runtime_state_timestamp(text)
    if ts is None:
        return False, (
            f"{deep_dive_path.name}: missing '## Runtime State — as of <timestamp>' "
            "section; add it or regenerate with generate_deep_dive.py"
        )

    if handoff_cutoff.tzinfo is None:
        handoff_cutoff = handoff_cutoff.replace(tzinfo=dt.UTC)

    if ts < handoff_cutoff:
        return False, (
            f"{deep_dive_path.name}: runtime state is stale: "
            f"section timestamp {ts.isoformat()} < handoff cutoff {handoff_cutoff.isoformat()}"
        )

    return True, (
        f"{deep_dive_path.name}: runtime state is fresh "
        f"({ts.isoformat()} >= {handoff_cutoff.isoformat()})"
    )


__all__: list[str] = [
    "CATEGORY_WEIGHTS",
    "PR_PATTERNS",
    "TICKET_RE",
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
]

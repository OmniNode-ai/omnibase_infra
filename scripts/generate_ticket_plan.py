#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.25.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Generate Ticket Plan by querying Linear API dynamically.

Design goals:
- Fully dynamic: NO hardcoded ticket lists
- Fetches all issues from "Beta Demo - January 2026" project
- Checks actual blocking relations from Linear
- Detects PR attachments to distinguish "in progress" from "ready"
- Categorizes into: READY TO WORK ON, IN PROGRESS, BLOCKED, IN REVIEW

Usage:
  uv run tools/generate_ticket_plan.py
  uv run tools/generate_ticket_plan.py --out /path/to/output.md
  uv run tools/generate_ticket_plan.py --dry-run  # Print to stdout
  uv run tools/generate_ticket_plan.py --project "My Project"  # Different project

Environment:
  LINEAR_API_KEY - Your Linear API key (from ~/.env or environment)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# =============================================================================
# CONSTANTS
# =============================================================================

LINEAR_API_URL = "https://api.linear.app/graphql"
DEFAULT_PROJECT = "Beta Demo - January 2026"

# Repo detection from labels (case-insensitive)
REPO_LABEL_MAP: dict[str, str] = {
    "omnimemory": "omnimemory",
    "omnidash": "omnidash",
    "omnibase_infra": "omnibase_infra",
    "omnibase_core": "omnibase_core",
    "omnibase_spi": "omnibase_spi",
    "omniintelligence": "omniintelligence",
    "omniclaude": "omniclaude",
    "infrastructure": "omnibase_infra",
    "frontend": "omnidash",
    "dashboard": "omnidash",
    "memory": "omnimemory",
    "intelligence": "omniintelligence",
    "claude": "omniclaude",
}

# Status type mappings (all lowercase for comparison)
DONE_STATUS_TYPES = {
    "completed",
    "canceled",
    "cancelled",
    "done",
    "closed",
    "duplicate",
}
IN_PROGRESS_STATUS_TYPES = {"started", "in progress", "in_progress"}
IN_REVIEW_STATUS_TYPES = {"in review", "in_review", "review"}
BACKLOG_STATUS_TYPES = {"backlog", "todo", "triage", "unstarted"}

# Priority names (Linear uses 0-4, 0=none, 1=urgent, 4=low)
PRIORITY_NAMES = {
    0: "None",
    1: "Urgent",
    2: "High",
    3: "Medium",
    4: "Low",
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BlockingRelation:
    """A blocking relation between issues."""

    blocker_id: str
    blocker_identifier: str
    blocker_status: str
    blocker_status_type: str


@dataclass
class Attachment:
    """An attachment on an issue (may be a PR)."""

    title: str
    url: str

    @property
    def is_github_pr(self) -> bool:
        """Check if this is a GitHub PR link."""
        return bool(re.match(r"https://github\.com/.+/pull/\d+", self.url))

    @property
    def pr_number(self) -> str | None:
        """Extract PR number if this is a GitHub PR."""
        match = re.search(r"/pull/(\d+)", self.url)
        return match.group(1) if match else None


@dataclass
class LinearIssue:
    """Parsed Linear issue data with relations."""

    id: str
    identifier: str
    title: str
    status: str
    status_type: (
        str  # Linear's state type: backlog, unstarted, started, completed, canceled
    )
    priority: int | None
    priority_name: str
    labels: list[str]
    attachments: list[Attachment]
    blocked_by: list[BlockingRelation] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)  # Issue identifiers this blocks

    @property
    def is_done(self) -> bool:
        """Check if issue is completed/canceled."""
        return (
            self.status_type.lower() in DONE_STATUS_TYPES
            or self.status.lower() in DONE_STATUS_TYPES
        )

    @property
    def is_in_progress(self) -> bool:
        """Check if issue is in progress."""
        return (
            self.status_type.lower() in IN_PROGRESS_STATUS_TYPES
            or self.status.lower() in IN_PROGRESS_STATUS_TYPES
        )

    @property
    def is_in_review(self) -> bool:
        """Check if issue is in review."""
        return (
            self.status_type.lower() in IN_REVIEW_STATUS_TYPES
            or self.status.lower() in IN_REVIEW_STATUS_TYPES
        )

    @property
    def is_backlog(self) -> bool:
        """Check if issue is in backlog/unstarted status."""  # stub-ok
        return (
            self.status_type.lower() in BACKLOG_STATUS_TYPES
            or self.status.lower() in BACKLOG_STATUS_TYPES
        )

    @property
    def has_open_pr(self) -> bool:
        """Check if issue has a GitHub PR attached."""
        return any(a.is_github_pr for a in self.attachments)

    @property
    def pr_url(self) -> str | None:
        """Get the first PR URL if any."""
        for a in self.attachments:
            if a.is_github_pr:
                return a.url
        return None

    @property
    def pr_display(self) -> str:
        """Get PR display string (e.g., '#123')."""
        for a in self.attachments:
            if a.is_github_pr and a.pr_number:
                return f"#{a.pr_number}"
        return ""

    @property
    def has_unresolved_blockers(self) -> bool:
        """Check if any blockers are not done."""
        for blocker in self.blocked_by:
            if blocker.blocker_status_type.lower() not in DONE_STATUS_TYPES:
                if blocker.blocker_status.lower() not in DONE_STATUS_TYPES:
                    return True
        return False

    @property
    def unresolved_blockers(self) -> list[BlockingRelation]:
        """Get list of blockers that are not done."""
        return [
            b
            for b in self.blocked_by
            if b.blocker_status_type.lower() not in DONE_STATUS_TYPES
            and b.blocker_status.lower() not in DONE_STATUS_TYPES
        ]

    def get_repo(self) -> str:
        """Detect repo from labels."""
        for label in self.labels:
            label_lower = label.lower()
            for key, repo in REPO_LABEL_MAP.items():
                if key in label_lower:
                    return repo
        return "unknown"


# =============================================================================
# CATEGORIZATION
# =============================================================================


@dataclass
class CategorizedIssues:
    """Issues categorized by workability status."""

    ready_to_work: list[LinearIssue] = field(default_factory=list)
    in_progress_with_pr: list[LinearIssue] = field(default_factory=list)
    blocked: list[LinearIssue] = field(default_factory=list)
    in_review: list[LinearIssue] = field(default_factory=list)
    done: list[LinearIssue] = field(default_factory=list)


def categorize_issues(issues: list[LinearIssue]) -> CategorizedIssues:
    """
    Categorize issues into workability groups.

    Categories:
    - READY TO WORK ON: Backlog/Unstarted, all blockers done (or no blockers), no open PR
    - IN PROGRESS (has open PR): In Progress status with PR attached
    - BLOCKED: Has blockers that are NOT done
    - IN REVIEW: In Review status OR has PR ready to merge
    - DONE: Completed/Canceled (excluded from main output)
    """
    result = CategorizedIssues()

    for issue in issues:
        # Skip done issues for main categorization
        if issue.is_done:
            result.done.append(issue)
            continue

        # Check if blocked (has unresolved blockers)
        if issue.has_unresolved_blockers:
            result.blocked.append(issue)
            continue

        # In Review - either status or has PR
        if issue.is_in_review:
            result.in_review.append(issue)
            continue

        # In Progress with PR
        if issue.is_in_progress and issue.has_open_pr:
            result.in_progress_with_pr.append(issue)
            continue

        # In Progress without PR (treat as in progress with PR section for visibility)
        if issue.is_in_progress:
            result.in_progress_with_pr.append(issue)
            continue

        # Ready to work on - backlog/todo, no blockers, no PR
        if issue.is_backlog and not issue.has_open_pr:
            result.ready_to_work.append(issue)
            continue

        # Default: put in ready if not categorized
        result.ready_to_work.append(issue)

    # Sort each category by priority (lower number = higher priority)
    def sort_key(i: LinearIssue) -> tuple[int, str]:
        return (i.priority if i.priority else 99, i.identifier)

    result.ready_to_work.sort(key=sort_key)
    result.in_progress_with_pr.sort(key=sort_key)
    result.blocked.sort(key=sort_key)
    result.in_review.sort(key=sort_key)

    return result


# =============================================================================
# LINEAR API CLIENT
# =============================================================================


def get_api_key() -> str:
    """Get Linear API key from environment."""
    # Try loading from ~/.env first
    env_file = Path.home() / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Also try local .env
    local_env = Path.cwd() / ".env"
    if local_env.exists():
        load_dotenv(local_env)

    key = os.environ.get("LINEAR_API_KEY")
    if not key:
        print(
            "Error: LINEAR_API_KEY not found in environment or ~/.env", file=sys.stderr
        )
        print("Get your API key from: https://linear.app/settings/api", file=sys.stderr)
        sys.exit(1)
    return key


def fetch_project_issues(api_key: str, project_name: str) -> list[LinearIssue]:
    """
    Fetch all issues from a Linear project with blocking relations and attachments.

    Uses pagination to get all issues.
    """
    # GraphQL query with relations and attachments
    query = """
    query GetProjectIssues($projectName: String!, $cursor: String) {
      issues(
        filter: {
          project: { name: { eq: $projectName } }
        }
        first: 100
        after: $cursor
      ) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          identifier
          title
          state {
            name
            type
          }
          priority
          labels {
            nodes {
              name
            }
          }
          attachments {
            nodes {
              title
              url
            }
          }
          relations {
            nodes {
              type
              relatedIssue {
                id
                identifier
                title
                state {
                  name
                  type
                }
              }
            }
          }
        }
      }
    }
    """

    all_issues: list[LinearIssue] = []
    cursor: str | None = None

    with httpx.Client(
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        timeout=60.0,
    ) as client:
        while True:
            variables: dict[str, Any] = {"projectName": project_name}
            if cursor:
                variables["cursor"] = cursor

            try:
                response = client.post(
                    LINEAR_API_URL,
                    json={"query": query, "variables": variables},
                )
                response.raise_for_status()
                data = response.json()

                if "errors" in data:
                    print(f"GraphQL errors: {data['errors']}", file=sys.stderr)
                    break

                issues_data = data.get("data", {}).get("issues", {})
                nodes = issues_data.get("nodes", [])

                for node in nodes:
                    issue = parse_issue(node)
                    if issue:
                        all_issues.append(issue)

                # Check pagination
                page_info = issues_data.get("pageInfo", {})
                if page_info.get("hasNextPage"):
                    cursor = page_info.get("endCursor")
                else:
                    break

            except httpx.HTTPError as e:
                print(f"HTTP error fetching issues: {e}", file=sys.stderr)
                break
            except Exception as e:  # noqa: BLE001 — boundary: prints error and degrades
                print(f"Error fetching issues: {e}", file=sys.stderr)
                break

    return all_issues


def parse_issue(node: dict[str, Any]) -> LinearIssue | None:
    """Parse a Linear issue node into a LinearIssue object."""
    try:
        state = node.get("state", {})
        priority = node.get("priority")

        # Parse labels
        labels = [label["name"] for label in node.get("labels", {}).get("nodes", [])]

        # Parse attachments
        attachments = [
            Attachment(title=a.get("title", ""), url=a.get("url", ""))
            for a in node.get("attachments", {}).get("nodes", [])
        ]

        # Parse relations
        blocked_by: list[BlockingRelation] = []
        blocks: list[str] = []

        for rel in node.get("relations", {}).get("nodes", []):
            rel_type = rel.get("type", "").lower().replace("_", " ").replace("-", " ")
            related = rel.get("relatedIssue", {})

            if not related:
                continue

            related_state = related.get("state", {})

            # Handle various relation type formats from Linear API
            # "blocks" type means this issue blocks the related issue
            # "blocked by" / "is blocked by" type means the related issue blocks this one
            if rel_type == "blocks":
                blocks.append(related.get("identifier", ""))
            elif rel_type in ("blocked by", "is blocked by", "blockedby"):
                blocked_by.append(
                    BlockingRelation(
                        blocker_id=related.get("id", ""),
                        blocker_identifier=related.get("identifier", ""),
                        blocker_status=related_state.get("name", "Unknown"),
                        blocker_status_type=related_state.get("type", "unknown"),
                    )
                )

        return LinearIssue(
            id=node.get("id", ""),
            identifier=node.get("identifier", ""),
            title=node.get("title", ""),
            status=state.get("name", "Unknown"),
            status_type=state.get("type", "unknown"),
            priority=priority,
            priority_name=PRIORITY_NAMES.get(priority, "None") if priority else "None",
            labels=labels,
            attachments=attachments,
            blocked_by=blocked_by,
            blocks=blocks,
        )
    except Exception as e:  # noqa: BLE001 — boundary: prints error and degrades
        print(f"Error parsing issue: {e}", file=sys.stderr)
        return None


# =============================================================================
# DOCUMENT GENERATION
# =============================================================================


def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def generate_document(categorized: CategorizedIssues, project_name: str) -> str:
    """Generate the ticket plan document."""
    lines: list[str] = []
    today = dt.datetime.now(tz=dt.UTC).date().strftime("%Y-%m-%d")

    # Header
    lines.append(f"# Ticket Plan - {today}")
    lines.append("")
    lines.append(f"**Project**: {project_name}")
    lines.append(f"**Generated**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary stats
    total_active = (
        len(categorized.ready_to_work)
        + len(categorized.in_progress_with_pr)
        + len(categorized.blocked)
        + len(categorized.in_review)
    )
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| Ready to Work On | {len(categorized.ready_to_work)} |")
    lines.append(f"| In Progress | {len(categorized.in_progress_with_pr)} |")
    lines.append(f"| Blocked | {len(categorized.blocked)} |")
    lines.append(f"| In Review | {len(categorized.in_review)} |")
    lines.append(f"| **Total Active** | **{total_active}** |")
    lines.append(f"| Done | {len(categorized.done)} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # READY TO WORK ON
    lines.append("## READY TO WORK ON")
    lines.append("")
    lines.append("*No blockers, no open PR - can start immediately*")
    lines.append("")

    if categorized.ready_to_work:
        lines.append("| Ticket | Title | Repo | Priority |")
        lines.append("|--------|-------|------|----------|")
        for issue in categorized.ready_to_work:
            title = truncate(issue.title, 45)
            repo = issue.get_repo()
            priority = issue.priority_name
            lines.append(f"| **{issue.identifier}** | {title} | {repo} | {priority} |")
    else:
        lines.append("*No tickets ready to work on*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # IN PROGRESS (has open PR)
    lines.append("## IN PROGRESS")
    lines.append("")
    lines.append("*Currently being worked on*")
    lines.append("")

    if categorized.in_progress_with_pr:
        lines.append("| Ticket | Title | Repo | PR | Status |")
        lines.append("|--------|-------|------|-------|--------|")
        for issue in categorized.in_progress_with_pr:
            title = truncate(issue.title, 40)
            repo = issue.get_repo()
            pr = issue.pr_display if issue.has_open_pr else "-"
            status = issue.status
            lines.append(
                f"| **{issue.identifier}** | {title} | {repo} | {pr} | {status} |"
            )
    else:
        lines.append("*No tickets in progress*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # BLOCKED
    lines.append("## BLOCKED")
    lines.append("")
    lines.append("*Has unresolved blockers - cannot start until blockers are done*")
    lines.append("")

    if categorized.blocked:
        lines.append("| Ticket | Title | Blocked By | Blocker Status |")
        lines.append("|--------|-------|------------|----------------|")
        for issue in categorized.blocked:
            title = truncate(issue.title, 35)
            blockers = issue.unresolved_blockers
            if blockers:
                blocker_ids = ", ".join(b.blocker_identifier for b in blockers)
                blocker_statuses = ", ".join(b.blocker_status for b in blockers)
            else:
                blocker_ids = "-"
                blocker_statuses = "-"
            lines.append(
                f"| {issue.identifier} | {title} | {blocker_ids} | {blocker_statuses} |"
            )
    else:
        lines.append("*No blocked tickets*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # IN REVIEW
    lines.append("## IN REVIEW")
    lines.append("")
    lines.append("*Ready for review or awaiting merge*")
    lines.append("")

    if categorized.in_review:
        lines.append("| Ticket | Title | Repo | PR |")
        lines.append("|--------|-------|------|-------|")
        for issue in categorized.in_review:
            title = truncate(issue.title, 40)
            repo = issue.get_repo()
            pr = issue.pr_display if issue.has_open_pr else "-"
            lines.append(f"| {issue.identifier} | {title} | {repo} | {pr} |")
    else:
        lines.append("*No tickets in review*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Blocking graph (what unlocks what)
    lines.append("## Blocking Graph")
    lines.append("")
    lines.append("*When these complete, they unlock other tickets*")
    lines.append("")

    # Find issues that block others
    blockers_map: dict[str, list[str]] = {}
    all_issues = (
        categorized.ready_to_work
        + categorized.in_progress_with_pr
        + categorized.blocked
        + categorized.in_review
        + categorized.done
    )

    for issue in all_issues:
        if issue.blocks:
            blockers_map[issue.identifier] = issue.blocks

    if blockers_map:
        lines.append("| When Complete | Unlocks |")
        lines.append("|---------------|---------|")
        for blocker_id, unlocks in sorted(blockers_map.items()):
            unlocks_str = ", ".join(unlocks)
            lines.append(f"| {blocker_id} | {unlocks_str} |")
    else:
        lines.append("*No blocking relationships found*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Legend
    lines.append("## Legend")
    lines.append("")
    lines.append("**Categories**:")
    lines.append(
        "- **READY TO WORK ON**: Status is Backlog/Todo, all blockers are Done (or no blockers), no open PR"
    )
    lines.append("- **IN PROGRESS**: Status is In Progress")
    lines.append("- **BLOCKED**: Has blockers that are NOT Done")
    lines.append("- **IN REVIEW**: Status is In Review or has PR ready to merge")
    lines.append("")
    lines.append("**Priority Levels**:")
    lines.append("- Urgent (1) - Critical path, immediate attention")
    lines.append("- High (2) - Important, should be done soon")
    lines.append("- Medium (3) - Normal priority")
    lines.append("- Low (4) - Nice to have")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate Ticket Plan from Linear (fully dynamic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tools/generate_ticket_plan.py
  uv run tools/generate_ticket_plan.py --project "My Project"
  uv run tools/generate_ticket_plan.py --dry-run
  uv run tools/generate_ticket_plan.py --out /path/to/output.md
        """,
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path. Defaults to TICKET_PLAN.md in current directory",
    )
    ap.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"Linear project name to query. Default: '{DEFAULT_PROJECT}'",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout instead of writing to file",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress",
    )
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path.cwd() / "TICKET_PLAN.md"

    print(f"Fetching issues from project: {args.project}", file=sys.stderr)

    api_key = get_api_key()

    print("Querying Linear API...", file=sys.stderr)
    issues = fetch_project_issues(api_key, args.project)

    if not issues:
        print(f"Warning: No issues found in project '{args.project}'", file=sys.stderr)
        print(
            "Check that the project name is correct and you have access.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(issues)} issues", file=sys.stderr)

    if args.verbose:
        for issue in issues:
            blockers = [b.blocker_identifier for b in issue.blocked_by]
            pr = issue.pr_display if issue.has_open_pr else "-"
            print(
                f"  {issue.identifier}: {issue.status} (blockers: {blockers}, PR: {pr})",
                file=sys.stderr,
            )

    print("Categorizing issues...", file=sys.stderr)
    categorized = categorize_issues(issues)

    print(f"  Ready: {len(categorized.ready_to_work)}", file=sys.stderr)
    print(f"  In Progress: {len(categorized.in_progress_with_pr)}", file=sys.stderr)
    print(f"  Blocked: {len(categorized.blocked)}", file=sys.stderr)
    print(f"  In Review: {len(categorized.in_review)}", file=sys.stderr)
    print(f"  Done: {len(categorized.done)}", file=sys.stderr)

    print("Generating document...", file=sys.stderr)
    document = generate_document(categorized, args.project)

    if args.dry_run:
        print(document)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(document)
        print(f"Wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

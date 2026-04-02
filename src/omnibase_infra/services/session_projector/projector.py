# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session projector — pure projection functions for agent session snapshots."""

from __future__ import annotations

import re
from datetime import datetime


def project_session_started(
    *,
    agent_id: str,
    session_id: str,
    terminal_id: str | None,
    machine: str | None,
    working_directory: str,
    git_branch: str | None,
    started_at: datetime,
) -> dict[str, object]:
    """Create initial snapshot from session-started event."""
    ticket = _extract_ticket_from_branch(git_branch or "")
    return {
        "agent_id": agent_id,
        "session_id": session_id,
        "terminal_id": terminal_id,
        "machine": machine,
        "working_directory": working_directory,
        "git_branch": git_branch,
        "current_ticket": ticket,
        "files_touched": [],
        "errors_hit": [],
        "last_tool_name": None,
        "last_tool_success": None,
        "last_tool_summary": None,
        "session_outcome": None,
        "session_started_at": started_at,
        "session_ended_at": None,
        "snapshot_at": started_at,
    }


def project_tool_executed(
    *,
    snapshot: dict[str, object],
    tool_name: str,
    success: bool,
    summary: str | None,
) -> dict[str, object]:
    """Update snapshot from tool-executed event."""
    updated = dict(snapshot)
    updated["last_tool_name"] = tool_name
    updated["last_tool_success"] = success
    updated["last_tool_summary"] = (summary or "")[:500]

    # Track files from Edit/Write tools
    # Summaries may contain absolute paths (/Volumes/..., C:\...) or
    # relative paths (src/foo.py:42). Match both forms.
    if tool_name in ("Edit", "Write") and summary:
        file_match = re.search(
            r"(/[^\s:]+|[A-Za-z]:\\[^\s:]+|[a-zA-Z_][a-zA-Z0-9_./\\-]+\.[a-zA-Z]{1,10})",
            summary,
        )
        if file_match:
            files = list(updated.get("files_touched") or [])
            path = file_match.group(1).rstrip(":")
            if path not in files:
                files.append(path)
            updated["files_touched"] = files

    # Track errors from failed tools
    if not success and summary:
        errors = list(updated.get("errors_hit") or [])
        errors.append(summary[:500])
        updated["errors_hit"] = errors[-20:]  # Keep last 20

    return updated


def project_session_ended(
    *,
    snapshot: dict[str, object],
    outcome: str,
    ended_at: datetime,
) -> dict[str, object]:
    """Finalize snapshot from session-ended event."""
    updated = dict(snapshot)
    updated["session_outcome"] = outcome
    updated["session_ended_at"] = ended_at
    updated["snapshot_at"] = ended_at
    return updated


def _extract_ticket_from_branch(branch: str) -> str | None:
    """Extract ticket ID from branch name."""
    match = re.search(r"[Oo][Mm][Nn]-(\d+)", branch)
    return f"OMN-{match.group(1)}" if match else None

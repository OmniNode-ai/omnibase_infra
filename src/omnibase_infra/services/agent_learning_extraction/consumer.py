# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Agent learning extraction consumer.

Listens to session-ended events on Kafka. For sessions with outcome SUCCESS,
extracts structured learning records by:
1. Collecting related tool-executed events from a short-lived buffer
2. Extracting error signatures from failed tool outputs
3. Generating a resolution summary via Qwen3-14B
4. Building a ModelAgentLearning record for storage
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from omnibase_infra.models.agent_learning.enum_learning_task_type import (
    EnumLearningTaskType,
)
from omnibase_infra.models.agent_learning.model_agent_learning import (
    ModelAgentLearning,
)

# Known repo directories in omni_home and omni_worktrees
_KNOWN_REPOS = {
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
    "omnibase_compat",
}

_TICKET_PATTERN = re.compile(r"[Oo][Mm][Nn]-(\d+)")

_CI_FILE_PATTERNS = {
    ".github/",
    "pyproject.toml",
    "pre-commit",
    "ruff",
    "ci.yml",
    "ci.yaml",
}
_MIGRATION_FILE_PATTERNS = {"migrations/", ".sql"}
_TEST_FILE_PATTERNS = {"tests/", "test_", "_test.py"}
_DOCS_FILE_PATTERNS = {"docs/", ".md", "README", "CLAUDE.md"}
_DEPENDENCY_FILE_PATTERNS = {
    "requirements",
    "pyproject.toml",
    "package.json",
    "uv.lock",
}


def extract_repo_from_working_dir(working_dir: str) -> str:
    """Extract repository name from a working directory path."""
    parts = working_dir.rstrip("/").split("/")
    for part in reversed(parts):
        if part in _KNOWN_REPOS:
            return part
    return "unknown"


def extract_ticket_from_branch(branch: str) -> str | None:
    """Extract ticket ID (e.g., OMN-7100) from a git branch name."""
    match = _TICKET_PATTERN.search(branch)
    if match:
        return f"OMN-{match.group(1)}"
    return None


def classify_task_type(
    branch: str,
    file_paths: list[str],
) -> EnumLearningTaskType:
    """Classify the task type from branch name and touched file paths."""
    all_text = " ".join(file_paths).lower() + " " + branch.lower()

    if any(p in all_text for p in _CI_FILE_PATTERNS):
        return EnumLearningTaskType.CI_FIX
    if any(p in all_text for p in _MIGRATION_FILE_PATTERNS):
        return EnumLearningTaskType.MIGRATION
    if any(p in all_text for p in _TEST_FILE_PATTERNS):
        return EnumLearningTaskType.TEST
    if any(p in all_text for p in _DOCS_FILE_PATTERNS):
        return EnumLearningTaskType.DOCS
    if any(p in all_text for p in _DEPENDENCY_FILE_PATTERNS):
        return EnumLearningTaskType.DEPENDENCY
    if "refactor" in all_text:
        return EnumLearningTaskType.REFACTOR
    if "fix" in all_text or "bug" in all_text:
        return EnumLearningTaskType.BUG_FIX

    return EnumLearningTaskType.FEATURE


def extract_error_signatures(
    tool_events: list[dict[str, object]],
) -> list[str]:
    """Extract error messages from failed tool execution events."""
    errors: list[str] = []
    for event in tool_events:
        if not event.get("success", True) and event.get("summary"):
            summary = str(event["summary"]).strip()
            if summary:
                errors.append(summary)
    return errors


def build_learning_record(
    *,
    session_id: UUID,
    working_dir: str,
    branch: str,
    resolution_summary: str,
    file_paths: list[str],
    error_signatures: list[str],
    created_at: datetime,
) -> ModelAgentLearning:
    """Build a structured learning record from session data."""
    repo = extract_repo_from_working_dir(working_dir)
    ticket_id = extract_ticket_from_branch(branch)
    task_type = classify_task_type(branch=branch, file_paths=file_paths)

    return ModelAgentLearning(
        session_id=session_id,
        repo=repo,
        file_paths_touched=tuple(file_paths),
        error_signatures=tuple(error_signatures),
        resolution_summary=resolution_summary,
        ticket_id=ticket_id,
        task_type=task_type,
        created_at=created_at,
    )

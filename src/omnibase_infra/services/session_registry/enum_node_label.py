# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node label enum for session coordination graph.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 8).
"""

from __future__ import annotations

from enum import StrEnum


class EnumNodeLabel(StrEnum):
    """Node labels for the session coordination graph.

    Values:
        SESSION: A Claude Code agent session.
        TASK: A Linear ticket or pipeline task being worked on.
        FILE: A source file touched by a task.
        PULL_REQUEST: A GitHub pull request produced by a task.
        REPOSITORY: A git repository that files and PRs belong to.
    """

    SESSION = "Session"
    TASK = "Task"
    FILE = "File"
    PULL_REQUEST = "PullRequest"
    REPOSITORY = "Repository"

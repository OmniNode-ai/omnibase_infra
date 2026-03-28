# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Relationship type enum for session coordination graph.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 8).
"""

from __future__ import annotations

from enum import StrEnum


class EnumRelationshipType(StrEnum):
    """Relationship types for the session coordination graph.

    Values:
        WORKS_ON: Session -> Task. A session is actively working on a task.
        TOUCHES: Task -> File. A task modifies or reads a file.
        DEPENDS_ON: Task -> Task. A task depends on another task.
        PRODUCED: Task -> PullRequest. A task produced a pull request.
        BELONGS_TO: File -> Repository or PullRequest -> Repository.
    """

    WORKS_ON = "WORKS_ON"
    TOUCHES = "TOUCHES"
    DEPENDS_ON = "DEPENDS_ON"
    PRODUCED = "PRODUCED"
    BELONGS_TO = "BELONGS_TO"

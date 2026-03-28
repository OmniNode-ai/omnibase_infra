# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session phase enum for session registry service.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 3).
"""

from __future__ import annotations

from enum import StrEnum


class EnumSessionPhase(StrEnum):
    """Lifecycle phase of a task being worked on across sessions.

    Ordering is significant for Doctrine D3 (replay-safe projectors):
    phases must only advance forward in this order, or use newest-event
    timestamp as tiebreaker.

    Values:
        PLANNING: Task is being planned or scoped.
        IMPLEMENTING: Code is being written.
        REVIEWING: Code review or local review in progress.
        MERGING: PR created, CI running, merge pending.
        DEPLOYING: Post-merge deployment in progress.
        COMPLETED: Task is done.
        STALLED: No activity for extended period.
    """

    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    REVIEWING = "reviewing"
    MERGING = "merging"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    STALLED = "stalled"

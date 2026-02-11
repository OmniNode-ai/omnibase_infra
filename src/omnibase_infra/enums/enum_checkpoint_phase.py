# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pipeline checkpoint phases for resume and replay.

Each phase represents a completed step in the ticket pipeline workflow.
Checkpoints are written after each phase completes (never during).

Ticket: OMN-2143
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumCheckpointPhase(str, Enum):
    """Pipeline phase that a checkpoint records.

    Attributes:
        IMPLEMENT: Code implementation completed.
        LOCAL_REVIEW: Local review iterations completed.
        CREATE_PR: Pull request created on remote.
        PR_RELEASE_READY: PR passed release-ready review.
        READY_FOR_MERGE: PR is ready for merge.
    """

    IMPLEMENT = "implement"
    """Code implementation phase completed."""

    LOCAL_REVIEW = "local_review"
    """Local review iterations completed."""

    CREATE_PR = "create_pr"
    """Pull request created on remote."""

    PR_RELEASE_READY = "pr_release_ready"
    """PR passed release-ready review."""

    READY_FOR_MERGE = "ready_for_merge"
    """PR is ready for merge."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    @property
    def phase_number(self) -> int:
        """Return the 1-based ordinal position of this phase."""
        ordered = list(EnumCheckpointPhase)
        return ordered.index(self) + 1

    @property
    def filename(self) -> str:
        """Return the canonical checkpoint filename for this phase."""
        return f"phase_{self.phase_number}_{self.value}.yaml"


__all__: list[str] = ["EnumCheckpointPhase"]

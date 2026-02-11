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
        """Return the stable 1-based ordinal for this phase.

        These ordinals are baked into checkpoint filenames on disk
        (``phase_{N}_{value}_a{attempt}.yaml``).  They MUST NOT change
        even if enum members are reordered or new members are inserted.
        """
        return _PHASE_ORDINALS[self]

    @property
    def filename(self) -> str:
        """Return the canonical checkpoint filename for this phase."""
        return f"phase_{self.phase_number}_{self.value}.yaml"


# Explicit mapping — add new phases at the end with the next available ordinal.
_PHASE_ORDINALS: dict[EnumCheckpointPhase, int] = {
    EnumCheckpointPhase.IMPLEMENT: 1,
    EnumCheckpointPhase.LOCAL_REVIEW: 2,
    EnumCheckpointPhase.CREATE_PR: 3,
    EnumCheckpointPhase.PR_RELEASE_READY: 4,
    EnumCheckpointPhase.READY_FOR_MERGE: 5,
}


__all__: list[str] = ["EnumCheckpointPhase"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Immutable package provenance against a detached registry mirror (OMN-17427)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["ModelOmnimarketRegistryStamp"]


@dataclass(frozen=True)
class ModelOmnimarketRegistryStamp:
    """Keep executed package, fetched reference and unused checkout distinct."""

    installed_commit: str
    reference_commit: str
    reference_ref: str
    checkout_commit: str
    commits_behind: int
    stamped_at: datetime

    @property
    def line(self) -> str:
        """Render the same provenance carried by the run receipt."""
        return (
            "drift_guard: mode=registry-reference "
            f"installed={self.installed_commit[:12]} "
            f"reference={self.reference_commit[:12]} "
            f"ref={self.reference_ref} "
            f"checkout={self.checkout_commit[:12]} "
            f"behind={self.commits_behind} "
            "verdict=proceed reason=installed_commit_reachable_from_fetched_default"
        )

    def as_receipt_fields(self) -> dict[str, object]:
        """Return JSON-serializable facts about the code actually dispatched."""
        return {
            "mode": "registry-reference",
            "verdict": "proceed",
            "installed_commit": self.installed_commit,
            "reference_commit": self.reference_commit,
            "reference_ref": self.reference_ref,
            "checkout_commit": self.checkout_commit,
            "commits_behind": self.commits_behind,
            "stamped_at": self.stamped_at.isoformat(),
            "line": self.line,
        }

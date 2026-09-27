# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Who issued an ``onex delegate`` run: the caller's lane and session.

OMN-19860. The lab ``delegation_events`` projection recorded which model
answered a delegation and how it ended, but never who asked, so per-lane
delegation use could not be queried from the event stream. The CLI now states
the caller on the request: the ledger lane in the request ``metadata``, the
session in the request's declared ``session_id``. Each half carries the rule
that chose it, so the stderr line tells a stated lane from a derived one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateCaller"]


class ModelDelegateCaller(BaseModel):
    """The caller's lane and session, and which rule chose each."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: str | None = Field(
        default=None,
        description="The caller's ledger lane token; None when nothing named one",
    )
    lane_source: str = Field(
        default="none",
        description="The rule that chose the lane, plus any value it skipped",
    )
    session_id: str | None = Field(
        default=None,
        description="The caller's session as a canonical UUID string; None when none",
    )
    session_source: str = Field(
        default="none",
        description="The rule that chose the session, plus any value it skipped",
    )

    @classmethod
    def unattributed(cls) -> ModelDelegateCaller:
        """A caller nothing names: the request gains no attribution key."""
        return cls()

    def describe(self) -> str:
        """One stderr line naming both halves and how each was chosen."""
        return (
            f"caller: lane={self.lane or 'none'} ({self.lane_source}) "
            f"session={self.session_id or 'none'} ({self.session_source})"
        )

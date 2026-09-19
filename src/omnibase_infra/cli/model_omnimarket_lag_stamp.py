# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The stamp a dispatch carries when its omnimarket is a known ancestor of the
canonical clone head (OMN-18814).

Rendered once, consumed twice -- as a structured stderr line during the run and
as a receipt fragment that outlives the terminal -- from this one object, so the
two cannot disagree about how far behind the run was. Same shape and the same
two consumers as :class:`ModelOffRegistryCheck`, deliberately: a reader of a
receipt should not have to learn a second vocabulary to answer the same
question.

WHY A STAMP AND NOT A REFUSAL, in the one case this covers. The guard's job is
to keep an unverifiable build from producing something that later reads as
evidence. An installed commit that is a strict ANCESTOR of the clone head is
not unverifiable: those bytes are merged, reviewed and reachable from the head,
they are simply not the tip. Refusing them bought nothing and cost everything --
measured 2026-09-19, the plugin CLI venv was in a refusing state about 545 of
740 elapsed minutes, roughly 74% of the day, against 159 omnimarket dev commits
in 7 days where every single commit opens a window.

WHY EVERY FIELD IS REQUIRED. A stamp whose fields can be omitted is a stamp that
lies by omission. A receipt carrying no ``commits_behind`` reads, to an auditor
a month later, exactly like a run that was at the tip; one carrying no
``clone_head`` cannot be checked against anything. There is no sensible default
for "how far behind was this run" -- the only honest values are a measured
number or no stamp at all, and no stamp at all is what the equal-commit path
already returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["ModelOmnimarketLagStamp"]


@dataclass(frozen=True)
class ModelOmnimarketLagStamp:
    """One ancestor-lag verdict, and the line that states it.

    Every field is required and none carries a default. See the module
    docstring for why that is a property rather than an oversight.
    """

    installed_commit: str
    clone_head: str
    commits_behind: int
    stamped_at: datetime

    @property
    def line(self) -> str:
        """The one structured line, emitted whenever this stamp is minted."""
        return (
            "drift_guard: "
            "mode=ancestor-lag "
            f"installed={self.installed_commit[:12]} "
            f"clone_head={self.clone_head[:12]} "
            f"behind={self.commits_behind} "
            "verdict=proceed "
            "reason=installed_commit_is_a_known_ancestor_of_clone_head"
        )

    def as_receipt_fields(self) -> dict[str, object]:
        """The same facts, for a run receipt.

        ``stamped_at`` is rendered to an ISO-8601 string here rather than left
        as a ``datetime``: the receipt is serialized to JSON, and a type the
        serializer refuses would fail at write time -- after the dispatch, with
        the result already produced -- instead of being caught by this module's
        own tests.
        """
        return {
            "mode": "ancestor-lag",
            "verdict": "proceed",
            "installed_commit": self.installed_commit,
            "clone_head": self.clone_head,
            "commits_behind": self.commits_behind,
            "stamped_at": self.stamped_at.isoformat(),
            "line": self.line,
        }

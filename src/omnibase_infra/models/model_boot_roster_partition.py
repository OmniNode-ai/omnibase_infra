# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot roster partition report (OMN-18087)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelBootRosterPartition(BaseModel):
    """How the boot interleave accounted for every contract in the roster.

    The roster is the set of contracts the auto-wiring report names. Each one
    must be in exactly one place: attempted at boot (an attach result), or
    excluded before any attempt (an exclusion, ``DEMAND_DEFERRED`` among them).

    Attributes:
        roster: Every contract name in the auto-wiring report.
        attempted: Contracts the interleave attempted at boot.
        excluded: Contracts reported as excluded, ``DEMAND_DEFERRED`` included.
        unaccounted: Roster contracts in neither ``attempted`` nor ``excluded``.
        in_both: Contracts reported both attempted and excluded.
        foreign: Reported contracts that are not in the roster.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    roster: tuple[str, ...]
    attempted: tuple[str, ...]
    excluded: tuple[str, ...]
    unaccounted: tuple[str, ...]
    in_both: tuple[str, ...]
    foreign: tuple[str, ...]

    @property
    def is_partition(self) -> bool:
        """True when every roster contract sits in exactly one set."""
        return not (self.unaccounted or self.in_both or self.foreign)

    def describe_violations(self) -> str:
        """Name each violating contract, grouped by the way it violates."""
        return (
            f"unaccounted={list(self.unaccounted)} in_both={list(self.in_both)} "
            f"foreign={list(self.foreign)}"
        )


__all__: list[str] = ["ModelBootRosterPartition"]

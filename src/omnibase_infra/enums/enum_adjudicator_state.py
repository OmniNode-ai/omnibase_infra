# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Adjudicator FSM states for the validation pipeline."""

from __future__ import annotations

from enum import Enum


class EnumAdjudicatorState(str, Enum):
    """FSM states for the validation adjudicator reducer.

    State flow: COLLECTING -> ADJUDICATING -> VERDICT_EMITTED.

    Values:
        COLLECTING: Accumulating check results from executor.
        ADJUDICATING: Applying scoring policy to collected results.
        VERDICT_EMITTED: Final verdict has been produced.
    """

    COLLECTING = "collecting"
    """Accumulating check results from executor."""

    ADJUDICATING = "adjudicating"
    """Applying scoring policy to collected results."""

    VERDICT_EMITTED = "verdict_emitted"
    """Final verdict has been produced."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumAdjudicatorState"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""A/B run variant classification for baseline comparison infrastructure.

Identifies whether a run is the BASELINE (no pattern applied) or the
CANDIDATE (pattern applied) in an A/B comparison.
"""

from __future__ import annotations

from enum import Enum


class EnumRunVariant(str, Enum):
    """Variant label for a single run in an A/B baseline comparison.

    Values:
        BASELINE: Run executed without the pattern applied.
        CANDIDATE: Run executed with the pattern applied.
    """

    BASELINE = "baseline"
    """Run executed without the pattern applied."""

    CANDIDATE = "candidate"
    """Run executed with the pattern applied."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumRunVariant"]

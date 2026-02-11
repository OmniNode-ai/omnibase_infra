# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Check severity classification for validation pipeline checks."""

from __future__ import annotations

from enum import Enum


class EnumCheckSeverity(str, Enum):
    """Severity classification for validation checks.

    Determines whether a check failure blocks the verdict or is advisory.

    Values:
        REQUIRED: Failure causes FAIL verdict. Must pass for PASS.
        RECOMMENDED: Failure contributes to QUARANTINE scoring.
        INFORMATIONAL: Reported but does not affect verdict.
    """

    REQUIRED = "required"
    """Failure causes FAIL verdict."""

    RECOMMENDED = "recommended"
    """Failure contributes to QUARANTINE scoring."""

    INFORMATIONAL = "informational"
    """Reported but does not affect verdict."""

    def blocks_verdict(self) -> bool:
        """Return True if this severity blocks a PASS verdict on failure."""
        return self == EnumCheckSeverity.REQUIRED

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumCheckSeverity"]

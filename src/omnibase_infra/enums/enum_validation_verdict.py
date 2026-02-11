# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation verdict outcomes for the adjudicator pipeline."""

from __future__ import annotations

from enum import Enum


class EnumValidationVerdict(str, Enum):
    """Verdict outcomes emitted by the validation adjudicator.

    Precedence (highest to lowest): FAIL > QUARANTINE > PASS.

    Values:
        PASS: All required checks pass, score >= threshold.
        FAIL: Hard block — required checks failed, prohibited edits, missing artifacts.
        QUARANTINE: Soft block — flake suspected, diff too large, high-risk without evidence.
    """

    PASS = "pass"
    """All required checks pass, no block reasons, score >= threshold."""

    FAIL = "fail"
    """Hard block — tests fail, typecheck fails, prohibited edits, missing artifacts."""

    QUARANTINE = "quarantine"
    """Soft block — flake suspected, diff too large, high-risk without evidence."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumValidationVerdict"]

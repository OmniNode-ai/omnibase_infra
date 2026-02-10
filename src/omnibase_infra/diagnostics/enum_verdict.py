# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tiered health verdict enumeration for bus health diagnostics."""

from __future__ import annotations

from enum import Enum


class EnumVerdict(str, Enum):
    """Tiered health verdict for a topic or the overall audit.

    EnumVerdict precedence (highest to lowest): FAIL > WARN > PASS.

    Values:
        PASS: All checks passed.
        WARN: Non-critical issues detected (empty topics, naming violations).
        FAIL: Critical issues detected (missing topics, schema errors, high parse failure).
    """

    PASS = "pass"
    """All checks passed."""

    WARN = "warn"
    """Non-critical issues detected."""

    FAIL = "fail"
    """Critical issues detected."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumVerdict"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Violation severity enum for the architecture validator node.

This module defines the severity levels for architecture violations.
"""

from __future__ import annotations

from enum import Enum


class EnumViolationSeverity(str, Enum):
    """Severity level of an architecture violation.

    Attributes:
        ERROR: A violation that must be fixed before code can be merged.
            Examples: use of Any type, missing type annotations, etc.
        WARNING: A violation that should be addressed but is not blocking.
            Examples: missing docstring, suboptimal patterns, etc.
    """

    ERROR = "error"
    WARNING = "warning"


__all__ = ["EnumViolationSeverity"]

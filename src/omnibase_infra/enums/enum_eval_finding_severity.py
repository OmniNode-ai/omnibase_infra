# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Eval finding severity enum.

Related:
    - OMN-6795: Define eval task models and enums

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from enum import Enum


class EnumEvalFindingSeverity(str, Enum):
    """Severity level of an evaluation finding."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


__all__: list[str] = ["EnumEvalFindingSeverity"]

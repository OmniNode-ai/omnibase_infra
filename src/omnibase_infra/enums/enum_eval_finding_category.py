# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Eval finding category enum.

Related:
    - OMN-6795: Define eval task models and enums

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from enum import Enum


class EnumEvalFindingCategory(str, Enum):
    """Category of an evaluation finding."""

    BUG = "bug"
    TECH_DEBT = "tech_debt"
    DOC_STALE = "doc_stale"
    REGRESSION_RISK = "regression_risk"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"


__all__: list[str] = ["EnumEvalFindingCategory"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Eval task status enum.

Related:
    - OMN-6795: Define eval task models and enums

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from enum import Enum


class EnumEvalTaskStatus(str, Enum):
    """Status of an evaluation task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


__all__: list[str] = ["EnumEvalTaskStatus"]

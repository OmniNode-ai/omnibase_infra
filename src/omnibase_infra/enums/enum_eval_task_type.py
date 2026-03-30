# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Eval task type enum for autonomous off-peak evaluations.

Defines the types of evaluation tasks that can be dispatched to
cheap LLM providers (Gemini Flash, GLM, local small models) during
off-peak periods.

Related:
    - OMN-6795: Define eval task models and enums

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from enum import Enum


class EnumEvalTaskType(str, Enum):
    """Types of autonomous evaluation tasks.

    Each type maps to a specific prompt template and evaluation
    strategy in ``ServiceEvalRunner``.

    Attributes:
        CODE_REVIEW: Automated code review of recently changed files.
        TECH_DEBT_SCAN: Scan for tech debt patterns and anti-patterns.
        DOC_FRESHNESS: Check documentation freshness against code changes.
        REGRESSION_TEST: Run regression test analysis on changed paths.
        HOSTILE_REVIEW: Adversarial review looking for subtle bugs.
    """

    CODE_REVIEW = "code_review"
    TECH_DEBT_SCAN = "tech_debt_scan"
    DOC_FRESHNESS = "doc_freshness"
    REGRESSION_TEST = "regression_test"
    HOSTILE_REVIEW = "hostile_review"


__all__: list[str] = ["EnumEvalTaskType"]

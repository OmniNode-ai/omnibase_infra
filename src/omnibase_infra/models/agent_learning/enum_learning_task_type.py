# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Classification of the task an agent was performing."""

from __future__ import annotations

from enum import StrEnum


class EnumLearningTaskType(StrEnum):
    """Classification of the task an agent was performing."""

    CI_FIX = "ci_fix"
    MIGRATION = "migration"
    FEATURE = "feature"
    REFACTOR = "refactor"
    BUG_FIX = "bug_fix"
    TEST = "test"
    DOCS = "docs"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"

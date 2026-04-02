# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mergeability gate status enum."""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumMergeabilityStatus(str, Enum):
    """Mergeability gate verdict."""

    MERGEABLE = "mergeable"
    NEEDS_SPLIT = "needs-split"
    BLOCKED = "blocked"

    def __str__(self) -> str:
        return self.value

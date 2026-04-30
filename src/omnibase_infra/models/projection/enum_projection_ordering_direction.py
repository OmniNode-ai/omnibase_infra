# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection ordering direction enum."""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumProjectionOrderingDirection(str, Enum):
    """Supported projection ordering directions."""

    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"


__all__ = ["EnumProjectionOrderingDirection"]

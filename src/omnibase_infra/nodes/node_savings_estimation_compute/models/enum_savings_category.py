# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Savings category enum for pattern type classification.

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

from enum import StrEnum


class EnumSavingsCategory(StrEnum):
    """Pattern type categories for savings breakdown."""

    ARCHITECTURE = "architecture"
    FILE = "file"
    TOOL = "tool"


__all__: list[str] = ["EnumSavingsCategory"]

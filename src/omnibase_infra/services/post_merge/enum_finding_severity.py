# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Post-merge finding severity enumeration.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

from enum import StrEnum


class EnumFindingSeverity(StrEnum):
    """Severity levels for post-merge findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


__all__ = ["EnumFindingSeverity"]

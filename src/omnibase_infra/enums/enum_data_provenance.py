# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Data provenance enum for tracking origin quality of projected values."""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumDataProvenance(str, Enum):
    """Origin quality classification for data values displayed in projections."""

    DEMO_SEEDED = "demo_seeded"
    DEMO_PROJECTED_SHORTCUT = "demo_projected_shortcut"
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


__all__ = ["EnumDataProvenance"]

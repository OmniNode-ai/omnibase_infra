# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle observations recorded by the first-effect recording scaffold."""

from __future__ import annotations

from enum import Enum


class EnumFirstEffectLedgerState(str, Enum):
    """The only legal observed lifecycle states; they confer no permission."""

    ISSUED = "ISSUED"
    PREFLIGHT_CONSUMED = "PREFLIGHT_CONSUMED"
    PUBLISHING = "PUBLISHING"
    PUBLISHED_UNKNOWN = "PUBLISHED_UNKNOWN"
    TERMINAL_OBSERVED = "TERMINAL_OBSERVED"
    BLOCKED = "BLOCKED"


__all__: list[str] = ["EnumFirstEffectLedgerState"]

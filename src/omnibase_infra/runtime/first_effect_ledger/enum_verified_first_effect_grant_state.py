# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""States owned by the durable verified-grant boundary."""

from __future__ import annotations

from enum import Enum


class EnumVerifiedFirstEffectGrantState(str, Enum):
    """Legal lifecycle separates publish ambiguity from consumer claim ownership."""

    VERIFIED = "VERIFIED"
    STAGED = "STAGED"
    PUBLISHING = "PUBLISHING"
    PUBLISHED_UNKNOWN = "PUBLISHED_UNKNOWN"
    CLAIMED = "CLAIMED"
    TERMINAL = "TERMINAL"


__all__ = ["EnumVerifiedFirstEffectGrantState"]

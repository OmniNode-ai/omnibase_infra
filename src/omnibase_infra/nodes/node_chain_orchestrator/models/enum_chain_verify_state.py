# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""FSM state enum for chain verification workflow."""

from __future__ import annotations

from enum import Enum


class EnumChainVerifyState(str, Enum):
    """States for the chain learning verification FSM."""

    PENDING = "pending"
    RETRIEVING = "retrieving"
    REPLAYING = "replaying"
    EXPLORING = "exploring"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FALLBACK = "fallback"
    FAILED = "failed"


__all__ = ["EnumChainVerifyState"]

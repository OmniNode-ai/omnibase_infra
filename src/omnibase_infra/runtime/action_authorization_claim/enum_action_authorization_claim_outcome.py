# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Closed result vocabulary for durable action-authorization nonce claims."""

from __future__ import annotations

from enum import StrEnum


class EnumActionAuthorizationClaimOutcome(StrEnum):
    """The only public outcomes from an attempted pre-execution claim."""

    CLAIMED = "CLAIMED"
    ALREADY_CONSUMED = "ALREADY_CONSUMED"
    EXPIRED = "EXPIRED"
    MISMATCH = "MISMATCH"
    ERROR = "ERROR"


__all__ = [
    "EnumActionAuthorizationClaimOutcome",
]

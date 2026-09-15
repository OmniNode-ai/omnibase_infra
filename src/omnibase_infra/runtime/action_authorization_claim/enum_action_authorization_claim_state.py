# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable state vocabulary for action-authorization nonce claims."""

from __future__ import annotations

from enum import StrEnum


class EnumActionAuthorizationClaimState(StrEnum):
    """The durable lifecycle retained by the dedicated claim table."""

    CLAIMED = "CLAIMED"
    EXPIRED = "EXPIRED"


__all__ = ["EnumActionAuthorizationClaimState"]

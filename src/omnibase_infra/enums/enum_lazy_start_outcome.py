# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How the lazy-start front door answered a held request (OMN-21592)."""

from __future__ import annotations

from enum import StrEnum

__all__: list[str] = ["EnumLazyStartOutcome"]


class EnumLazyStartOutcome(StrEnum):
    """DELIVERED, or a typed reason the request was not answered by the node."""

    DELIVERED = "DELIVERED"
    START_TIMEOUT = "START_TIMEOUT"
    START_FAILED = "START_FAILED"
    CONTRACT_NOT_ANNOUNCED = "CONTRACT_NOT_ANNOUNCED"

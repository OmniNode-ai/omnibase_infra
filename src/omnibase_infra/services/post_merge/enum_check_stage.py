# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Post-merge check stage enumeration.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

from enum import StrEnum


class EnumCheckStage(StrEnum):
    """Post-merge check stages."""

    HOSTILE_REVIEW = "hostile_review"
    CONTRACT_SWEEP = "contract_sweep"
    INTEGRATION_CHECK = "integration_check"


__all__ = ["EnumCheckStage"]

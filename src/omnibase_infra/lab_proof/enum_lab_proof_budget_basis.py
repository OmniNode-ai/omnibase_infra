# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Whether a profile budget was measured on a lab host or is an estimate.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofBudgetBasis(StrEnum):
    """Whether a profile budget was measured on a lab host or is an estimate."""

    MEASURED = "measured"
    ESTIMATE = "estimate"


__all__ = ["EnumLabProofBudgetBasis"]

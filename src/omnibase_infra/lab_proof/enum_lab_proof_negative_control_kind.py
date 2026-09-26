# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""How a profile shows that its own checks can fail.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofNegativeControlKind(StrEnum):
    """How a profile shows that its own checks can fail."""

    SABOTAGE_IMPORT = "sabotage_import"
    BASE_RERUN = "base_rerun"
    NONE = "none"


__all__ = ["EnumLabProofNegativeControlKind"]

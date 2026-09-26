# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""What runs a profile: its declared nodes, a hand-run recipe, or nothing.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofExecution(StrEnum):
    """What runs a profile: its declared nodes, a hand-run recipe, or nothing."""

    NODE = "node"
    MANUAL_RECIPE = "manual_recipe"
    NONE = "none"


__all__ = ["EnumLabProofExecution"]

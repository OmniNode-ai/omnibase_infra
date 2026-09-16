# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Repository visibility, as it bears on placement.

Ticket: OMN-18412
"""

from __future__ import annotations

from enum import StrEnum


class EnumCIRunnerRouteVisibility(StrEnum):
    """Repository visibility, as it bears on placement.

    ``INTERNAL`` is deliberately absent: it is not publicly readable and its
    hosted minutes are billed exactly like a private repository's, so it
    normalises to ``PRIVATE``. Treating it as public because the API spells it
    differently would be a silent exemption from the ruling.
    """

    PUBLIC = "public"
    PRIVATE = "private"
    UNKNOWN = "unknown"


__all__ = ["EnumCIRunnerRouteVisibility"]

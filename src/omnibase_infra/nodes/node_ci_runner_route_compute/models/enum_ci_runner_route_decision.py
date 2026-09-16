# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""What the router decided about a run's placement.

Ticket: OMN-18412
"""

from __future__ import annotations

from enum import StrEnum


class EnumCIRunnerRouteDecision(StrEnum):
    """What the router decided."""

    SELF_HOSTED = "self_hosted"
    HOSTED = "hosted"
    # No placement is allowed. Not a crash and not a fallback: the only
    # placement the policy permits is one the repository may not use.
    BLOCKED = "blocked"


__all__ = ["EnumCIRunnerRouteDecision"]

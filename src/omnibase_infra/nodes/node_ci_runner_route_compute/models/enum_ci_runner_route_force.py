# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Operator override for the routing decision.

Ticket: OMN-18412
"""

from __future__ import annotations

from enum import StrEnum


class EnumCIRunnerRouteForce(StrEnum):
    """Operator override.

    An override is subject to every safety rule, not a bypass of them: FLEET
    still cannot widen past the seam ceiling and still cannot take a fork pull
    request, and HOSTED is still refused for a private repository. It exists so
    the kill switch the static seam provided survives the move to per-run
    routing.
    """

    AUTO = "auto"
    FLEET = "fleet"
    HOSTED = "hosted"


__all__ = ["EnumCIRunnerRouteForce"]

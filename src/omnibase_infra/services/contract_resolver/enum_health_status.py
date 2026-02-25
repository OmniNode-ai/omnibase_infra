# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Health status enum for the Contract Resolver Bridge.

Ticket: OMN-2756
"""

from __future__ import annotations

from enum import Enum


class EnumHealthStatus(str, Enum):
    """Service health status values for the Contract Resolver Bridge."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


__all__ = ["EnumHealthStatus"]

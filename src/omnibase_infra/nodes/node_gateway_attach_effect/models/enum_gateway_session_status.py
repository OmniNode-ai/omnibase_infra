# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Attach session lifecycle states."""

from __future__ import annotations

from enum import Enum


class EnumGatewaySessionStatus(str, Enum):
    """Lifecycle states of one attach session."""

    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    DETACHED = "DETACHED"
    REVOKED = "REVOKED"


__all__ = ["EnumGatewaySessionStatus"]

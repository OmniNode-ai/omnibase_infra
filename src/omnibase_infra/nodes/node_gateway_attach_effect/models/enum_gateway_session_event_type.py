# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session lifecycle event types."""

from __future__ import annotations

from enum import Enum


class EnumGatewaySessionEventType(str, Enum):
    ATTACHED = "ATTACHED"
    HEARTBEAT_OK = "HEARTBEAT_OK"
    HEARTBEAT_DEGRADED = "HEARTBEAT_DEGRADED"
    REVOKED = "REVOKED"
    DETACHED = "DETACHED"


__all__ = ["EnumGatewaySessionEventType"]

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
    # OMN-16022: the two enforced-teardown outcomes. Both are emitted by
    # the heartbeat handler, which is the only path that both notices the
    # bound and can return a thin-publish payload for the projection.
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"


__all__ = ["EnumGatewaySessionEventType"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a heartbeat tore its session down.

OMN-16022 replaced ``ModelGatewayHeartbeatResponse.revoked: bool`` with a
nullable member of this enum. The boolean could only ever express one of
the three terminal outcomes the heartbeat path now has, and a second and
third boolean beside it would have left callers to reconstruct a
three-way choice from a flag soup. ``None`` means the session survived the
heartbeat.
"""

from __future__ import annotations

from enum import Enum


class EnumGatewaySessionTerminationReason(str, Enum):
    """Terminal outcomes of one heartbeat call."""

    # Keycloak introspection returned a clean 200 saying ``active: false``
    # (or naming a different client). Real revocation -- never an outage.
    REVOKED = "REVOKED"
    # ``now >= session.expires_at``: the session outlived the ceiling set
    # at attach as min(token exp, max_session_ttl_seconds).
    EXPIRED = "EXPIRED"
    # The session went longer than ``max_unverified_session_seconds``
    # without a successful revalidation. Bounded degraded mode: the point
    # at which "we cannot currently check this credential" stops being a
    # reason to keep trusting it.
    UNVERIFIED_CEILING = "UNVERIFIED_CEILING"


__all__ = ["EnumGatewaySessionTerminationReason"]

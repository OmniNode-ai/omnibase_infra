# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Attach session lifecycle states."""

from __future__ import annotations

from enum import Enum


class EnumGatewaySessionStatus(str, Enum):
    """Lifecycle states of one attach session.

    ``EXPIRED`` and ``QUARANTINED`` are both terminal and both mean "this
    session is gone, re-attach" -- they differ only in which bound fired
    (OMN-16022). ``REVOKED`` remains reserved for a Keycloak introspection
    that actually said ``active: false``; neither of the two new states is
    ever produced by an outage, which is the OMN-15918 invariant.
    """

    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    DETACHED = "DETACHED"
    REVOKED = "REVOKED"
    # The session outlived its stored ``expires_at`` (itself
    # min(token exp, max_session_ttl_seconds), set at attach).
    EXPIRED = "EXPIRED"
    # The session spent longer than ``max_unverified_session_seconds``
    # without a successful Keycloak revalidation and was torn down rather
    # than left alive un-revalidated for the duration of an outage.
    QUARANTINED = "QUARANTINED"


__all__ = ["EnumGatewaySessionStatus"]

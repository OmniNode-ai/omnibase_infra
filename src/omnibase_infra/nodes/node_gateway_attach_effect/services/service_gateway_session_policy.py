# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session lifetime policy -- the two bounds every session-consuming path applies.

OMN-16022. ``expires_at`` was written at attach and read by nothing, and
the OMN-15918 outage/revocation split (an unreachable Keycloak must never
be read as revocation) had no time bound composed with it. The result was
that a session could be revalidated forever past its own stored ceiling,
and could survive un-revalidated for as long as an outage -- or an
attacker-induced gateway<->Keycloak partition -- lasted. An adversary who
could hold that partition open held revocation open with it, including for
a credential the operator had just rotated.

Both bounds live here, as pure functions over a session plus a caller
supplied ``now``, for three reasons:

  - one definition, applied identically by heartbeat and detach, so a new
    session-consuming path cannot quietly acquire a third interpretation;
  - no I/O and no clock read of its own, so the handler tests drive the
    boundary directly instead of sleeping;
  - it stays out of ``handlers/`` because it performs no transport call,
    matching ``service_keycloak_token_validator``'s split of CPU-only
    verification from the handler-resident HTTP fetches.

The bounds are deliberately different things and must not be merged:
``expires_at`` bounds how long a session may live at all, and derives from
the token's own ``exp`` clamped by ``max_session_ttl_seconds`` (3600s).
The unverified ceiling bounds how long a session may live *without being
re-checked*, and derives from ``max_unverified_session_seconds`` (900s,
one attach-token lifetime). A session can breach either without the other.
"""

from __future__ import annotations

from datetime import datetime

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)


class SessionExpiredError(Exception):
    """Raised when a session-consuming path is handed a session past ``expires_at``."""


def is_expired(session: ModelGatewaySession, *, now: datetime) -> bool:
    """True once the session has reached the ceiling stamped on it at attach.

    Boundary is inclusive (``now >= expires_at``): at the instant the
    ceiling is reached the session is over. Fail-closed on the boundary is
    the cheap direction -- the remedy is a re-attach the runtime performs
    automatically.
    """
    return now >= session.expires_at


def unverified_seconds(session: ModelGatewaySession, *, now: datetime) -> float:
    """Seconds since this session was last SUCCESSFULLY validated.

    ``last_heartbeat_at`` carries that timestamp (see its field comment):
    attach seeds it, and only a heartbeat whose introspection returned
    active advances it. A heartbeat that arrives while Keycloak is
    unreachable does not, so this measures revalidation staleness and not
    request traffic.
    """
    return (now - session.last_heartbeat_at).total_seconds()


def exceeds_unverified_ceiling(
    session: ModelGatewaySession,
    *,
    now: datetime,
    config: ModelGatewayAttachConfig,
) -> bool:
    """True once the session has gone longer than the degraded-mode ceiling unverified."""
    return unverified_seconds(session, now=now) > config.max_unverified_session_seconds


__all__ = [
    "SessionExpiredError",
    "exceeds_unverified_ceiling",
    "is_expired",
    "unverified_seconds",
]

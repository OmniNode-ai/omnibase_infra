# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Renewal-cycle policy -- when the client must mint its NEXT session.

OMN-15952. Deliberately a separate module from
``service_gateway_session_policy``, which answers a different question about
a different subject:

  * ``service_gateway_session_policy`` bounds the life of the session that
    already exists -- has it passed ``expires_at``, has it gone too long
    unverified. Those are the server's decisions, applied on
    session-consuming paths, and their outcome is teardown.
  * this module computes the cycle the *client* runs so that a successor
    session exists before the incumbent one dies. Its outcome is a directive
    handed back at attach; it tears nothing down and reads no clock of its
    own.

Merging them would produce one module whose functions answer to two
authorities, and the OMN-15952 review is explicit that a second lifecycle
authority over one session is how a session ends up simultaneously live and
torn down. Keeping the split means the server's bounds cannot quietly become
client advice, or the reverse.

Everything here is a pure function over a session plus config, with ``now``
supplied by the caller: the boundary cases (a token so short-lived that the
margin already covers its whole life, a heartbeat exactly at the ceiling)
are then driven directly by tests instead of slept through.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_renewal_mode import (
    EnumGatewayRenewalMode,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_renewal_directive import (
    ModelGatewayRenewalDirective,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)


class ExpiryExtensionError(Exception):
    """Raised when a session revision would move ``expires_at``.

    The renewal contract's load-bearing negative: ``expires_at`` is stamped
    once at attach and is immutable for the life of that ``session_id``.
    Any path that produces a revised session must be able to prove it did
    not move the ceiling, which is what ``assert_expiry_not_extended``
    below is for.
    """


def build_renewal_directive(
    session: ModelGatewaySession, *, config: ModelGatewayAttachConfig
) -> ModelGatewayRenewalDirective:
    """Compute the renewal cycle for a freshly attached session.

    ``renew_at`` is ``expires_at - renewal_margin_seconds``; the jitter
    window opens ``renewal_jitter_seconds`` before that.

    Both are floored at ``attached_at``, which is what makes the function
    total rather than merely usually-correct. A session whose whole life is
    shorter than the margin (a token presented with 60 seconds left, or a
    deployment that shortens the token lifespan below the configured
    margin) would otherwise yield a ``renew_at`` in the past -- and the
    model would then reject the directive, turning an unusual-but-valid
    attach into a 5xx. Flooring says the honest thing instead: renew
    immediately, you have no headroom. The floor cannot collide with the
    model's strict ``renew_at < session_expires_at`` invariant, because
    ``attached_at < expires_at`` always holds (attach rejects a token with
    no remaining lifetime before a session is ever constructed).
    """
    renew_at = max(
        session.attached_at,
        session.expires_at - timedelta(seconds=config.renewal_margin_seconds),
    )
    renew_not_before = max(
        session.attached_at,
        renew_at - timedelta(seconds=config.renewal_jitter_seconds),
    )
    return ModelGatewayRenewalDirective(
        mode=EnumGatewayRenewalMode.RE_ATTACH,
        session_expires_at=session.expires_at,
        renew_not_before=renew_not_before,
        renew_at=renew_at,
        margin_seconds=config.renewal_margin_seconds,
        jitter_seconds=config.renewal_jitter_seconds,
    )


def is_renewal_due(
    session: ModelGatewaySession,
    *,
    now: datetime,
    config: ModelGatewayAttachConfig,
) -> bool:
    """True once the runtime is inside the window where it must renew.

    Boundary is inclusive at ``renew_not_before``: at the instant the window
    opens, renewal is due. Erring early is the cheap direction -- an early
    re-grant costs one token, whereas a late one costs the session.
    """
    directive = build_renewal_directive(session, config=config)
    return now >= directive.renew_not_before


def assert_expiry_not_extended(
    previous: ModelGatewaySession, revised: ModelGatewaySession
) -> None:
    """Guard that a session revision left the attach-time ceiling alone.

    This is the executable form of the contract's central negative. It is
    cheap to state and easy to violate by accident: every session revision
    in this node is a ``model_copy(update=...)``, and adding one key to that
    dict is all it takes to turn a heartbeat into a lifetime extension. The
    resulting defect would be invisible in every existing assertion (status
    and timestamps would all look right) and would silently reintroduce the
    in-place renewal the design refuses.

    Raises ``ExpiryExtensionError`` on any change to ``expires_at`` -- in
    either direction. Shortening is not a safe subset: it is still a second
    authority mutating a field whose whole value is that exactly one write,
    at attach, ever happens.
    """
    if revised.expires_at != previous.expires_at:
        raise ExpiryExtensionError(
            f"session {previous.session_id} expires_at moved from "
            f"{previous.expires_at.isoformat()} to {revised.expires_at.isoformat()}; "
            "the attach-time ceiling is immutable -- renewal is re-grant + "
            "re-attach (EnumGatewayRenewalMode.RE_ATTACH), never extension"
        )


__all__ = [
    "ExpiryExtensionError",
    "assert_expiry_not_extended",
    "build_renewal_directive",
    "is_renewal_due",
]

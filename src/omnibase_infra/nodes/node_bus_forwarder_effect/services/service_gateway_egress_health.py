# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Egress-denial accounting and its health verdict (OMN-17201).

Two halves that deliberately live in one file because they are one contract:
``publish_egress_health`` writes the counters and ``evaluate_egress_health``
reads them back through ``load_egress_health``. The writer runs inside the
forwarder process; the reader runs inside the container healthcheck. Splitting
them would let the file format drift between the only two things that touch it.

Free functions over a stateful class on purpose: the counters belong to
``NodeGatewayDelivery``, which is the canonical NODE that owns the delivery
lifecycle. A second lifecycle-owning object beside it would be exactly the
non-canonical shape the OMN-14350 ratchet exists to keep out.

Why this exists at all: quarantining an authorization-denied record keeps the
bridge moving, which is strictly better than wedging it -- but a leg where
EVERY record is denied then drains its lag into a dead letter queue and looks
indistinguishable from a healthy leg. That is a worse failure than the wedge,
because the wedge at least showed up as climbing lag. The counter is the price
of the quarantine.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayEgressHealth,
)

logger = logging.getLogger(__name__)

DEFAULT_EGRESS_HEALTH_PATH = "/tmp/gateway-egress-health.json"  # noqa: S108 -- container-local scratch state, same convention as the canary probe's --state-file
"""Shared default so the writer and the healthcheck cannot drift apart.

Container-local, not the ``/app/data`` named volume, and deliberately so:
these counters describe the CURRENT process. Surviving a restart would let a
denial burst from a previous process report the new one unhealthy before it
has published anything of its own.
"""

EGRESS_DENIAL_WINDOW_SECONDS = 120
"""How long a leg must go denial-only before the container reads unhealthy.

Four times the default 30s canary cadence. The window has to be wider than one
cadence for a reason that is not cosmetic: a leg carrying a MIX of denied and
permitted topics is degraded, not dead, and the operator action (grant one
ACL) is different from the dead case. Comparing only "last denial newer than
last delivery" would flip such a leg between healthy and unhealthy on
alternating records. Requiring a full window with denials and no delivery at
all reports exactly the condition worth paging on -- nothing is crossing --
and catches it within two 15s healthcheck ticks once it starts.
"""


def record_denial(
    state: ModelGatewayEgressHealth,
    *,
    topic: str,
    tenant_id: str,
    principal_id: str,
    now: datetime,
) -> ModelGatewayEgressHealth:
    """Return ``state`` with one destination-authorization denial accounted."""
    return state.model_copy(
        update={
            "denied_total": state.denied_total + 1,
            "last_denied_at": now,
            "last_denied_topic": topic,
            "last_denied_tenant_id": tenant_id,
            "last_denied_principal_id": principal_id,
        }
    )


def record_delivery(
    state: ModelGatewayEgressHealth,
    *,
    now: datetime,
) -> ModelGatewayEgressHealth:
    """Return ``state`` with one record that actually crossed accounted."""
    return state.model_copy(
        update={
            "delivered_total": state.delivered_total + 1,
            "last_delivered_at": now,
        }
    )


def publish_egress_health(
    state: ModelGatewayEgressHealth,
    state_path: Path | None,
) -> None:
    """Write the counters where the container healthcheck can read them.

    Best-effort by design, and never raises. This is an observability surface
    reached from the delivery path; a failed write must not turn a delivery
    that actually succeeded into an exception, because that would reintroduce
    -- from the monitoring code -- the class of wedge OMN-17201 removes. A
    ``None`` path counts in memory and publishes nothing, which is what an
    embedder without a health surface wants.
    """
    if state_path is None:
        return
    temporary = state_path.with_name(f"{state_path.name}.tmp")
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(state.model_dump_json(), encoding="utf-8")
        temporary.replace(state_path)
    except OSError:
        logger.warning(
            "Gateway egress health state write failed path=%s",
            state_path,
            exc_info=True,
        )


def load_egress_health(state_path: Path) -> ModelGatewayEgressHealth | None:
    """Read the counters the forwarder published, or ``None`` if unreadable.

    ``None`` means "no verdict available", never "healthy" and never
    "unhealthy" -- the caller decides. A missing file is the normal state of a
    container that has just started and has not yet forwarded or denied
    anything.
    """
    try:
        raw = state_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ModelGatewayEgressHealth.model_validate_json(raw)
    except ValueError:
        logger.warning("Gateway egress health state is unparseable path=%s", state_path)
        return None


def evaluate_egress_health(
    state: ModelGatewayEgressHealth | None,
    *,
    now: datetime,
    window_seconds: int = EGRESS_DENIAL_WINDOW_SECONDS,
) -> tuple[bool, str]:
    """Return ``(passed, detail)`` for the egress leg of the container health check.

    Fails only on the condition that is genuinely indistinguishable from an
    outage: denials inside the window and not one record delivered inside that
    same window. Every other shape passes, and says why -- a mixed leg reports
    its denial count on the PASS line so a partial ACL gap is still visible in
    ``docker inspect ...State.Health.Log`` rather than being rounded to "fine".

    Absence is not failure. ``state is None`` (no file yet) passes: a fresh
    container that has not forwarded anything has not proven anything wrong,
    and failing closed here would make the health check depend on the writer
    having run, which is the sentinel-file coupling OMN-15741 removed.
    """
    if state is None:
        return True, "egress leg: no denial state published yet"
    if state.last_denied_at is None:
        return (
            True,
            f"egress leg: no authorization denial recorded "
            f"(delivered={state.delivered_total})",
        )
    denial_age = (now - state.last_denied_at).total_seconds()
    if denial_age >= window_seconds:
        return (
            True,
            f"egress leg: last authorization denial was {denial_age:.0f}s ago, "
            f"outside the {window_seconds}s window "
            f"(denied={state.denied_total} delivered={state.delivered_total})",
        )
    delivered_recently = (
        state.last_delivered_at is not None
        and (now - state.last_delivered_at).total_seconds() < window_seconds
    )
    if delivered_recently:
        return (
            True,
            f"egress leg: degraded but crossing -- {state.denied_total} record(s) "
            f"denied, last on topic {state.last_denied_topic} for principal "
            f"{state.last_denied_principal_id}; records are still being "
            f"delivered (delivered={state.delivered_total})",
        )
    return (
        False,
        f"egress leg: every outbound record is being denied -- {state.denied_total} "
        f"record(s) quarantined and NOTHING delivered in the last "
        f"{window_seconds}s. Last denial {denial_age:.0f}s ago on topic "
        f"{state.last_denied_topic} for tenant_id={state.last_denied_tenant_id} "
        f"principal_id={state.last_denied_principal_id}. Grant the topic ACL for "
        f"that principal, or detach the tenant.",
    )


__all__ = [
    "DEFAULT_EGRESS_HEALTH_PATH",
    "EGRESS_DENIAL_WINDOW_SECONDS",
    "evaluate_egress_health",
    "load_egress_health",
    "publish_egress_health",
    "record_delivery",
    "record_denial",
]

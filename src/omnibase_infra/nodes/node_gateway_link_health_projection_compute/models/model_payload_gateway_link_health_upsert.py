# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelPayloadGatewayLinkHealthUpsert - intent payload for link-health persistence.

The COMPUTE node emits a ModelIntent carrying this payload after parsing a
``onex.evt.omnibase-infra.gateway-heartbeat.v1`` event (``ModelGatewayHeartbeat``,
published by ``node_bus_forwarder_effect``'s ``ServiceGatewayForwarder.publish_heartbeat``).
The EFFECT node consumes the intent and upserts one row into
``gateway_link_health``, keyed by ``tenant_id`` -- the row is never deleted, so
absence of a *fresh* heartbeat shows up as a stale ``last_seen_at`` on an
existing row (evaluated live by the ``gateway_link_health_status`` view), not
as a missing row.

SCOPE DISCLOSURE (OMN-15570 / G3, gateway lift Phase 0): this payload carries
only what ``ModelGatewayHeartbeat`` publishes today -- ``tenant_id``,
``principal_id``, ``emitted_at``, ``local_transport_flavor``. The contract's
``gateway_forwarder.liveness`` block (``node_bus_forwarder_effect/contract.yaml``
lines 59-63) also declares ``lag_threshold_messages`` /
``lag_threshold_seconds``, but no producer in this codebase publishes lag
data on the heartbeat topic (G1/G2 -- OMN-15741/OMN-15742 -- are the lanes
that would add reconnect/lag telemetry; both have since merged, but G2 added
``status`` / ``consecutive_failures`` / ``detail`` to the heartbeat rather
than lag fields, and this payload does not yet carry those either).
``lag_messages`` / ``lag_seconds`` are therefore always ``None`` on
this payload; the write-effect persists them as nullable columns and the
status view only evaluates the silence-window threshold it can actually
observe. See the module docstring on the status view for the exact
consequence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadGatewayLinkHealthUpsert(BaseModel):
    """Payload for a single gateway_link_health upsert.

    Attributes:
        intent_type: Discriminator literal, always
            "gateway_link_health.upsert".
        tenant_id: Tenant slug identifying the edge (heartbeat's
            ``tenant_id``, config-bound DNS-safe slug -- the current link
            identity granularity; one row per tenant edge).
        principal_id: Canonical MSK principal (``t-<tenant UUID hex>``) that
            emitted the heartbeat.
        local_transport_flavor: Local leg transport flavor
            ("containerized" | "lightweight") from the heartbeat.
        last_seen_at: Producer-supplied ``emitted_at`` timestamp of the
            heartbeat -- the freshness stamp the status view diffs against
            ``NOW()``.
        lag_messages: Consumer lag in messages, when a producer supplies it.
            Always ``None`` today -- see module docstring.
        lag_seconds: Consumer lag in seconds, when a producer supplies it.
            Always ``None`` today -- see module docstring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["gateway_link_health.upsert"] = Field(
        default="gateway_link_health.upsert",
        description="Discriminator literal for intent routing.",
    )

    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Tenant slug identifying the gateway edge (link identity).",
    )
    principal_id: str = Field(
        ...,
        min_length=1,
        description="Canonical MSK principal that emitted the heartbeat.",
    )
    local_transport_flavor: str = Field(
        ...,
        min_length=1,
        description="Local leg transport flavor from the heartbeat.",
    )
    last_seen_at: datetime = Field(
        ...,
        description="Producer-supplied heartbeat emission time.",
    )
    lag_messages: int | None = Field(
        default=None,
        description="Consumer lag in messages, when supplied. None today.",
    )
    lag_seconds: float | None = Field(
        default=None,
        description="Consumer lag in seconds, when supplied. None today.",
    )


__all__ = ["ModelPayloadGatewayLinkHealthUpsert"]

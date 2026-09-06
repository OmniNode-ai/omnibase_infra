# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-process egress-denial counters for the gateway container healthcheck.

OMN-17201. Quarantining an authorization-denied record (see
``GatewayEgressDeniedError``) stops the wedge, but on its own it converts a
loud failure -- lag climbing, records retained -- into a silent one: lag drains
because every record is dead-lettered, and the container keeps reporting
healthy. This model is the counter that makes the silent case observable.

It is a FILE contract, not just an in-process struct, because the two
processes that need it are different: the long-running forwarder writes it,
and the Docker healthcheck (``onex-gateway-canary-probe``, a separate exec in
the same container) reads it. The probe deliberately never talks to the
forwarder process -- see ``runtime/gateway_canary_probe.py`` -- so a file in
the shared container filesystem is the seam.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayEgressHealth(BaseModel):
    """Denial and delivery counters for one gateway process lifetime.

    Counters are lifetime-of-process, not windowed. The window belongs to the
    reader (``evaluate_egress_health``), which needs the raw timestamps to
    tell "everything is being denied right now" apart from "one topic was
    denied an hour ago and traffic has flowed since".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    denied_total: int = Field(default=0, ge=0)
    delivered_total: int = Field(default=0, ge=0)
    last_denied_at: datetime | None = Field(default=None)
    last_denied_topic: str | None = Field(default=None)
    last_denied_tenant_id: str | None = Field(default=None)
    last_denied_principal_id: str | None = Field(default=None)
    last_delivered_at: datetime | None = Field(default=None)


__all__ = ["ModelGatewayEgressHealth"]

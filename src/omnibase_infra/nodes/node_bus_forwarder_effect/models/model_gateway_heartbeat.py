# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed tenant-edge liveness event emitted by the gateway forwarder."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayHeartbeat(BaseModel):
    """Observable proof of one tenant-scoped edge's liveness/reconnect state.

    ``status="degraded"`` is emitted by the runtime reconnect-supervision
    loop (``runtime/gateway_forwarder.py``) once a cloud-leg delivery
    failure has persisted past the contract-declared
    ``degraded_after_seconds`` window; ``consecutive_failures``/``detail``
    are only populated on that path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Heartbeat tenant_id is the config-bound DNS-safe slug.
    tenant_id: str
    # Canonical MSK principal is t-<tenant UUID hex>, not a UUID field.
    principal_id: str
    status: Literal["active", "degraded"] = "active"
    emitted_at: datetime
    local_transport_flavor: Literal["containerized", "lightweight"]
    consecutive_failures: int = Field(default=0, ge=0)
    detail: str = ""


__all__ = ["ModelGatewayHeartbeat"]

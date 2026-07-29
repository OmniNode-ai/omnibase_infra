# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed tenant-edge liveness event emitted by the gateway forwarder."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelGatewayHeartbeat(BaseModel):
    """Observable proof that one tenant-scoped edge is attached and active."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Heartbeat tenant_id is the config-bound DNS-safe slug.
    tenant_id: str
    # Canonical MSK principal is t-<tenant UUID hex>, not a UUID field.
    principal_id: str
    status: Literal["active"] = "active"
    emitted_at: datetime
    local_transport_flavor: Literal["containerized", "lightweight"]


__all__ = ["ModelGatewayHeartbeat"]

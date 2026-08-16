# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session lifecycle event -- the thin-publish payload for the bus.

Handlers return this alongside their typed response; the node/runtime
owns the actual publish (node_owned_publish, mirroring
node_bus_forwarder_effect's capability of the same name) onto
the contract-declared gateway session-event topic. Handlers never call the bus
directly -- this keeps the effect boundary at the node, not scattered across
handler internals, and lets the eventual link-health projection (G3)
consume one canonical event shape for both the forwarder heartbeat and the
attach control plane.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_event_type import (
    EnumGatewaySessionEventType,
)


class ModelGatewaySessionEvent(BaseModel):
    """Session lifecycle fact, one per attach/heartbeat/detach/revoke."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: EnumGatewaySessionEventType
    session_id: UUID
    tenant_id: UUID
    tenant_slug: str
    principal_id: str
    edge_instance_id: str
    emitted_at: datetime


__all__ = ["ModelGatewaySessionEvent"]

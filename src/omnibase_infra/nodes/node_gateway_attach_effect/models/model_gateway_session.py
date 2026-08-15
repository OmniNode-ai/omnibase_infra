# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Attach session record -- the tenant-bound control-plane state unit."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)


class ModelGatewaySession(BaseModel):
    """Immutable-per-revision record of one tenant edge's attach session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    tenant_id: UUID
    tenant_slug: str
    principal_id: str
    # Keycloak client_id of the per-tenant confidential client, as bound at
    # attach time -- carried through heartbeats so the introspection re-check
    # always targets the right client without re-deriving it from the token.
    keycloak_client_id: str
    edge_instance_id: str
    status: EnumGatewaySessionStatus
    attached_at: datetime
    last_heartbeat_at: datetime
    # Never later than the underlying access token's exp claim, clamped by
    # ModelGatewayAttachConfig.max_session_ttl_seconds.
    expires_at: datetime


__all__ = ["ModelGatewaySession"]

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
    # Time of the last SUCCESSFUL validation of this session, not the time
    # the last heartbeat request arrived. It is seeded at attach (where the
    # token's signature and claims were verified against JWKS) and advanced
    # only by a heartbeat whose introspection returned active. A heartbeat
    # that arrives during a Keycloak outage deliberately leaves it
    # untouched -- that is what lets OMN-16022's
    # max_unverified_session_seconds ceiling measure real revalidation
    # staleness rather than mere request traffic, and it is why an attacker
    # who can partition gateway<->Keycloak cannot hold the clock still by
    # continuing to send heartbeats.
    last_heartbeat_at: datetime
    # Never later than the underlying access token's exp claim, clamped by
    # ModelGatewayAttachConfig.max_session_ttl_seconds. Enforced on every
    # session-consuming path since OMN-16022 (it was previously written at
    # attach and read by nothing).
    expires_at: datetime


__all__ = ["ModelGatewaySession"]

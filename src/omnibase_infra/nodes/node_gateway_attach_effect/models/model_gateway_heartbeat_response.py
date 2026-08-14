# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Heartbeat response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)


class ModelGatewayHeartbeatResponse(BaseModel):
    """Output of ``gateway.heartbeat``.

    ``revoked`` is True exactly when Keycloak introspection returned
    ``active: false`` for the presented token -- the session is torn down
    (status flips to REVOKED) in the same handler call, so a caller never
    observes a stale ACTIVE session after this response.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: ModelGatewaySession
    revoked: bool
    session_event: ModelGatewaySessionEvent


__all__ = ["ModelGatewayHeartbeatResponse"]

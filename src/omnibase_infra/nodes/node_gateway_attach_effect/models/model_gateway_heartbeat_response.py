# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Heartbeat response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_termination_reason import (
    EnumGatewaySessionTerminationReason,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)


class ModelGatewayHeartbeatResponse(BaseModel):
    """Output of ``gateway.heartbeat``.

    ``termination_reason`` is ``None`` exactly when the session survived
    this heartbeat. When it is set the session has already been removed
    from the store inside this same handler call and ``session`` carries
    the matching terminal status, so a caller never observes a stale
    ACTIVE session after this response.

    It replaced the previous ``revoked: bool`` in OMN-16022: the heartbeat
    path gained two further terminal outcomes (an enforced ``expires_at``
    and the bounded degraded-mode ceiling) and a boolean cannot name three
    outcomes. Note that only ``REVOKED`` means Keycloak actually said the
    credential was inactive -- an unreachable Keycloak still produces no
    termination at all below the ceiling (OMN-15918 R4).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: ModelGatewaySession
    termination_reason: EnumGatewaySessionTerminationReason | None
    session_event: ModelGatewaySessionEvent


__all__ = ["ModelGatewayHeartbeatResponse"]

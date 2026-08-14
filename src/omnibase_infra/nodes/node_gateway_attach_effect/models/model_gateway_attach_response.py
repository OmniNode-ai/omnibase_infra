# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Attach response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_renewal_directive import (
    ModelGatewayRenewalDirective,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)


class ModelGatewayAttachResponse(BaseModel):
    """Output of ``gateway.attach``.

    ``session_event`` is the thin-publish payload: the runtime publishes this
    node's output onto the contract-declared session-event topic, so the
    handler never calls the bus directly (node_owned_publish).

    ``renewal`` is the OMN-15952 addition and is REQUIRED, not optional. An
    unattended runtime that attaches and is not told the renewal cycle has
    no correct behaviour available to it -- it will either heartbeat into
    its own expiry or invent a policy. Making the field optional would let
    exactly that case ship silently, so a response without it does not
    validate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: ModelGatewaySession
    heartbeat_interval_seconds: int
    renewal: ModelGatewayRenewalDirective
    session_event: ModelGatewaySessionEvent


__all__ = ["ModelGatewayAttachResponse"]

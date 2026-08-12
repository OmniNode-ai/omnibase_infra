# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Attach response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

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
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: ModelGatewaySession
    heartbeat_interval_seconds: int
    session_event: ModelGatewaySessionEvent


__all__ = ["ModelGatewayAttachResponse"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Detach response."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)


class ModelGatewayDetachResponse(BaseModel):
    """Output of ``gateway.detach``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    status: EnumGatewaySessionStatus
    session_event: ModelGatewaySessionEvent


__all__ = ["ModelGatewayDetachResponse"]

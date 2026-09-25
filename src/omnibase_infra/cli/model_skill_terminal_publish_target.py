# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The broker a skill's terminal result is published to (OMN-19152)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)


class ModelSkillTerminalPublishTarget(BaseModel):
    """The broker a lane resolves to, and the identity to reach it as."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: str = Field(..., min_length=1)
    bootstrap_servers: str = Field(..., min_length=1)
    transport: ModelLaneClientTransport | None = Field(
        default=None,
        description="Resolved lane transport; None binds nothing.",
    )

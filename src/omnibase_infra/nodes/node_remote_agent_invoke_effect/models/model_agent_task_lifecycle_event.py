# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stub lifecycle event model for the remote-agent invoke effect node.

Full model definition deferred to OMN-9625/9626/9627 core delegation stack.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_remote_agent_invoke_effect.models.enum_agent_task_status import (
    EnumAgentTaskStatus,
)


class ModelAgentTaskLifecycleEvent(BaseModel):
    """Lifecycle event emitted while watching a remote agent task."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Tracks the originating invocation.")
    status: EnumAgentTaskStatus = Field(..., description="Task lifecycle status.")
    agent_id: UUID = Field(..., description="Agent that processed the task.")


__all__ = ["ModelAgentTaskLifecycleEvent"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Workspace pre-flight validate command (OMN-13247, plan §5.5).

Consumed by the pure COMPUTE node. Carries the policy inputs the deterministic
verdict needs. No git/clean-tree/HEAD-sha reads here — those are I/O and live in
the EFFECT. The verdict shape is ``ModelWorkspaceValidateResult``.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.enum_agent_sandbox import EnumAgentSandbox


class ModelWorkspaceValidateCommand(BaseModel):
    """Pure pre-flight inputs for the workspace COMPUTE node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow run correlation id.")
    workspace_path: str = Field(..., description="Requested workspace path.")
    allowed_roots: tuple[str, ...] = Field(
        ...,
        description="Allowed root prefixes; the resolved path must live under one.",
    )
    sandbox: EnumAgentSandbox = Field(..., description="Requested write posture.")
    prompt: str = Field(..., description="The task prompt (shape-validated).")


__all__: list[str] = ["ModelWorkspaceValidateCommand"]

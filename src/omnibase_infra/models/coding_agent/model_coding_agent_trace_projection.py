# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Correlation-trace projection for the coding-agent workflow.

Materialized by the FSM reducer (OMN-13247, plan §5.6). Mirrors the delegation
correlation-trace projection. Read via the generic projection API
(GET /projection/{topic}); never via a bespoke endpoint.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.enum_agent_sandbox import EnumAgentSandbox
from omnibase_infra.models.coding_agent.enum_cli_backend_status import (
    EnumCliBackendStatus,
)
from omnibase_infra.models.coding_agent.enum_coding_agent import EnumCodingAgent
from omnibase_infra.models.coding_agent.enum_coding_agent_fsm_state import (
    EnumCodingAgentFsmState,
)


class ModelCodingAgentTraceProjection(BaseModel):
    """One row of the coding-agent correlation-trace projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow run correlation id.")
    agent: EnumCodingAgent = Field(..., description="Target agent CLI.")
    workspace_hash: str = Field(
        default="",
        description="Hash of workspace_path (no raw path in the projection).",
    )
    sandbox: EnumAgentSandbox = Field(..., description="Write posture for this run.")
    status: EnumCodingAgentFsmState = Field(
        ..., description="Current FSM state (terminal once COMPLETED/FAILED/REJECTED)."
    )
    duration_ms: float = Field(default=0.0, description="Subprocess duration ms.")
    exit_code: int | None = Field(default=None, description="Subprocess exit code.")
    files_changed_count: int = Field(
        default=0, description="Count of git-derived changed files."
    )
    files_changed: tuple[str, ...] = Field(
        default_factory=tuple, description="git-derived changed file paths."
    )
    diff_hash: str = Field(default="", description="Hash of the captured diff.")
    output_hash: str = Field(default="", description="Hash of agent stdout.")
    error_class: EnumCliBackendStatus = Field(
        default=EnumCliBackendStatus.SUCCESS, description="Structured failure class."
    )
    escalation_source: str | None = Field(
        default=None,
        description="Set when invoked as the code_generation ceiling (Phase E).",
    )

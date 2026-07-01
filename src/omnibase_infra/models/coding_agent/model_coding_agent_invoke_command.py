# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed input command for the coding-agent workflow (OMN-13247, plan §5.4).

GRADUATION NOTE (memory ``feedback_models_in_core_not_market``): this command is
repo-private to ``omnibase_infra`` today. When Phase E wires the omnimarket
delegation orchestrator to emit this command at the code_generation ceiling, a
second repo will import it (>=2 importers) and it must graduate to
``omnibase_core``. Do NOT bump the ``omnibase_compat`` pin to host it.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.enum_agent_sandbox import EnumAgentSandbox
from omnibase_infra.models.coding_agent.enum_coding_agent import EnumCodingAgent


class ModelCodingAgentInvokeCommand(BaseModel):
    """One coding-agent invocation request.

    The client thin-publishes this command to the workflow's command topic
    (declared in the orchestrator contract); the orchestrator sequences
    validate -> invoke -> capture over the bus. ``model`` is the agent-native
    model id (NOT a ``*_MODEL`` env read); auth is ambient (no API key field).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Workflow run correlation id; dedupe key for the FSM."
    )
    agent: EnumCodingAgent = Field(
        ..., description="Which coding-agent CLI to invoke (CLAUDE | CODEX)."
    )
    prompt: str = Field(..., description="The task prompt handed to the agent.")
    workspace_path: str = Field(
        ...,
        description="Absolute path to the workspace the agent runs in; validated "
        "against allowed roots by the COMPUTE node before any subprocess.",
    )
    sandbox: EnumAgentSandbox = Field(
        ...,
        description="Write posture: READ_ONLY cannot edit; WORKSPACE_WRITE may.",
    )
    allow_dirty_tree: bool = Field(
        default=False,
        description="If False, a dirty git tree is rejected before invocation.",
    )
    model: str | None = Field(
        default=None,
        description="Agent-native model id (not a *_MODEL env read). None = "
        "the agent's own default.",
    )
    timeout_ms: int = Field(
        ..., gt=0, description="Hard wall-clock deadline for the subprocess, in ms."
    )
    network: bool = Field(
        default=False,
        description="Whether the agent may use the network. Default off.",
    )

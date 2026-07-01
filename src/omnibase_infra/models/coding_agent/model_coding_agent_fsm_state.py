# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""FSM state for the coding-agent workflow (OMN-13247, plan §5.2).

Owned and folded by ``node_coding_agent_fsm_reducer``. Pure data — the reducer
computes the next state from ``(state, event)``; it never performs I/O.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.enum_agent_sandbox import EnumAgentSandbox
from omnibase_infra.models.coding_agent.enum_coding_agent import EnumCodingAgent
from omnibase_infra.models.coding_agent.enum_coding_agent_fsm_state import (
    EnumCodingAgentFsmState,
)


class ModelCodingAgentFsmState(BaseModel):
    """Deterministic FSM state for one coding-agent invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow run correlation id.")
    agent: EnumCodingAgent = Field(..., description="Target agent CLI.")
    sandbox: EnumAgentSandbox = Field(..., description="Write posture for this run.")
    current_state: EnumCodingAgentFsmState = Field(
        default=EnumCodingAgentFsmState.IDLE, description="Current FSM state."
    )
    consecutive_failures: int = Field(
        default=0, description="Effect failures since the last success."
    )
    max_consecutive_failures: int = Field(
        default=3,
        gt=0,
        description="Circuit-breaker threshold; N failures -> FAILED.",
    )
    invoke_attempts: int = Field(
        default=0, description="Number of effect invocations issued for this run."
    )
    max_read_only_retries: int = Field(
        default=2,
        ge=0,
        description="Bounded retry budget for READ_ONLY invocations.",
    )
    error_message: str | None = Field(
        default=None, description="Last error message, if any."
    )

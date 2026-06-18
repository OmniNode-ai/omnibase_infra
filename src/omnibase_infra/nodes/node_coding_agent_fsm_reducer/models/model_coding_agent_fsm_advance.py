# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Advance command for the coding-agent FSM reducer (OMN-13247).

The reducer folds one workflow event into the FSM state. The prior state and the
event are the only inputs the pure transition needs. This node-local wrapper is
the reducer's dispatch input shape; the shared FSM/event models live in
``omnibase_infra.models.coding_agent``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.model_coding_agent_event import (
    ModelCodingAgentEvent,
)
from omnibase_infra.models.coding_agent.model_coding_agent_fsm_state import (
    ModelCodingAgentFsmState,
)


class ModelCodingAgentFsmAdvance(BaseModel):
    """One event folded into the coding-agent FSM state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ModelCodingAgentFsmState = Field(
        ..., description="Prior FSM state to advance from."
    )
    event: ModelCodingAgentEvent = Field(..., description="The event to fold.")


__all__: list[str] = ["ModelCodingAgentFsmAdvance"]

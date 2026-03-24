# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Frozen state model for the node graph reducer FSM [OMN-6349].

The reducer is stateless -- this model is the state passed in/out of reduce().
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeGraphState(BaseModel):
    """Immutable state for the node graph lifecycle FSM.

    FSM states: initializing -> wiring -> running -> draining -> stopped
    Wildcard: fatal_error transitions to stopped from any state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fsm_state: str = Field(
        default="initializing",
        description="Current FSM state of the node graph lifecycle.",
    )
    nodes_loaded: int = Field(
        default=0,
        description="Number of nodes loaded into the graph.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if in stopped state due to fatal_error.",
    )

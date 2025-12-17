# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM State Definition Model.

This module provides the ModelFSMStateDefinition for representing states
in the finite state machine.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.reducers.models.model_fsm_action import ModelFSMAction


class ModelFSMStateDefinition(BaseModel):
    """State definition for FSM.

    Represents a single state in the finite state machine, including
    entry and exit actions.

    Supports both 'state_name' (production format) and 'name' (simplified format)
    for backward compatibility with test fixtures.

    Attributes:
        state_name: Unique identifier for this state.
        description: Human-readable description of the state purpose.
        is_terminal: Whether this is a terminal (final) state.
        entry_actions: Actions to execute when entering this state.
        exit_actions: Actions to execute when leaving this state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str = Field(
        ...,
        alias="state_name",
        description="Unique identifier for this state",
    )
    description: str = Field(
        default="", description="Human-readable description of the state purpose"
    )
    is_terminal: bool = Field(
        default=False, description="Whether this is a terminal (final) state"
    )
    entry_actions: list[ModelFSMAction] = Field(
        default_factory=list,
        description="Actions to execute when entering this state",
    )
    exit_actions: list[ModelFSMAction] = Field(
        default_factory=list,
        description="Actions to execute when leaving this state",
    )


__all__ = ["ModelFSMStateDefinition"]

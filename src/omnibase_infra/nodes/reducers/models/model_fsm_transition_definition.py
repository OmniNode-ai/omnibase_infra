# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Transition Definition Model.

This module provides the ModelFSMTransitionDefinition for representing
state transitions in the finite state machine.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.reducers.models.model_fsm_action import ModelFSMAction
from omnibase_infra.nodes.reducers.models.model_fsm_condition import ModelFSMCondition


class ModelFSMTransitionDefinition(BaseModel):
    """Transition definition for FSM.

    Represents a state transition, including the trigger event,
    guard conditions, and actions to execute.

    Attributes:
        from_state: The source state name (using 'from' in YAML).
        to_state: The target state name (using 'to' in YAML).
        trigger: The event that triggers this transition.
        description: Human-readable description of the transition.
        conditions: Guard conditions that must be satisfied.
        actions: Actions to execute during the transition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    from_state: str = Field(..., alias="from", description="The source state name")
    to_state: str = Field(..., alias="to", description="The target state name")
    trigger: str = Field(..., description="The event that triggers this transition")
    description: str = Field(
        default="", description="Human-readable description of the transition"
    )
    conditions: list[ModelFSMCondition] = Field(
        default_factory=list,
        description="Guard conditions that must be satisfied",
    )
    actions: list[ModelFSMAction] = Field(
        default_factory=list,
        description="Actions to execute during the transition",
    )


__all__ = ["ModelFSMTransitionDefinition"]

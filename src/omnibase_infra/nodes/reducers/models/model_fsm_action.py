# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Action Model.

This module provides the ModelFSMAction for representing actions in FSM
state entry/exit or transitions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFSMAction(BaseModel):
    """Action definition for FSM state entry/exit or transition.

    Represents an action that can be executed during state transitions
    or when entering/exiting a state.

    Attributes:
        action: The action identifier/name to execute.
        description: Human-readable description of what the action does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: str = Field(..., description="The action identifier/name to execute")
    description: str = Field(
        default="", description="Human-readable description of what the action does"
    )


__all__ = ["ModelFSMAction"]

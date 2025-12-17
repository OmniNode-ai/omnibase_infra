# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Condition Model.

This module provides the ModelFSMCondition for representing guard conditions
in FSM transitions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFSMCondition(BaseModel):
    """Condition definition for FSM transitions.

    Represents a guard condition that must be satisfied for a transition
    to be allowed.

    Attributes:
        expression: The condition expression to evaluate.
        description: Human-readable description of the condition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    expression: str = Field(..., description="The condition expression to evaluate")
    description: str = Field(
        default="", description="Human-readable description of the condition"
    )


__all__ = ["ModelFSMCondition"]

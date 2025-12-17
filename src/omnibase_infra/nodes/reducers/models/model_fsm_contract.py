# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Contract Model.

This module provides the ModelFSMContract for representing the loaded
FSM contract definition from the YAML file.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.reducers.models.model_fsm_error_handling import (
    ModelFSMErrorHandling,
)
from omnibase_infra.nodes.reducers.models.model_fsm_state_definition import (
    ModelFSMStateDefinition,
)
from omnibase_infra.nodes.reducers.models.model_fsm_transition_definition import (
    ModelFSMTransitionDefinition,
)


class ModelFSMContract(BaseModel):
    """Loaded FSM contract definition.

    Represents the FSM contract loaded from
    contracts/fsm/dual_registration_reducer_fsm.yaml.

    Attributes:
        contract_version: Semantic version of the FSM contract.
        name: Name of the FSM workflow.
        description: Human-readable description of the FSM purpose.
        initial_state: Starting state for the FSM.
        states: List of state definitions from the contract.
        transitions: List of transition definitions from the contract.
        error_handling: Error handling configuration from the contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: str = Field(
        ..., description="Semantic version of the FSM contract"
    )
    name: str = Field(..., description="Name of the FSM workflow")
    description: str = Field(
        default="", description="Human-readable description of the FSM purpose"
    )
    initial_state: str = Field(..., description="Starting state for the FSM")
    states: list[ModelFSMStateDefinition] = Field(
        default_factory=list,
        description="List of state definitions from the contract",
    )
    transitions: list[ModelFSMTransitionDefinition] = Field(
        default_factory=list,
        description="List of transition definitions from the contract",
    )
    error_handling: ModelFSMErrorHandling = Field(
        default_factory=ModelFSMErrorHandling,
        description="Error handling configuration from the contract",
    )


__all__ = ["ModelFSMContract"]

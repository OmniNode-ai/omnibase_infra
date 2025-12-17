# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Contract Model for Dual Registration Workflow.

This module provides ModelFSMContract for representing the loaded FSM contract
definition from the YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelFSMContract:
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

    contract_version: str
    name: str
    description: str
    initial_state: str
    states: list[dict[str, object]] = field(default_factory=list)
    transitions: list[dict[str, object]] = field(default_factory=list)
    error_handling: dict[str, object] = field(default_factory=dict)


__all__ = ["ModelFSMContract"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validators for Architecture Validator node.

This module contains validator implementations for each architecture rule:
    - ARCH-001: No Direct Handler Dispatch validator
    - ARCH-002: No Handler Publishing Events validator
    - ARCH-003: No Workflow FSM in Orchestrators validator

Two interfaces are provided:

1. **Function-based validators** (legacy): Standalone functions for direct validation.
   - `validate_no_direct_dispatch(file_path)`
   - `validate_no_handler_publishing(file_path)`
   - `validate_no_orchestrator_fsm(file_path)`

2. **Protocol-compliant rule classes**: Implement `ProtocolArchitectureRule` for
   integration with the architecture validator framework.
   - `RuleNoDirectDispatch`
   - `RuleNoHandlerPublishing`
   - `RuleNoOrchestratorFSM`

Validators are wired through the contract.yaml configuration and follow
the detection_strategy patterns defined there.
"""

from omnibase_infra.nodes.architecture_validator.validators.validator_no_direct_dispatch import (
    RuleNoDirectDispatch,
    validate_no_direct_dispatch,
)
from omnibase_infra.nodes.architecture_validator.validators.validator_no_handler_publishing import (
    RuleNoHandlerPublishing,
    validate_no_handler_publishing,
)
from omnibase_infra.nodes.architecture_validator.validators.validator_no_orchestrator_fsm import (
    RuleNoOrchestratorFSM,
    validate_no_orchestrator_fsm,
)

__all__: list[str] = [
    # Functions (file-based validators)
    "validate_no_direct_dispatch",
    "validate_no_handler_publishing",
    "validate_no_orchestrator_fsm",
    # Classes (protocol-compliant rules)
    "RuleNoDirectDispatch",
    "RuleNoHandlerPublishing",
    "RuleNoOrchestratorFSM",
]

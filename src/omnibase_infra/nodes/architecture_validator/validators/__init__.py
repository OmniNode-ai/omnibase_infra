# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validators for Architecture Validator node.

This module provides validator implementations for ONEX architecture rules.
Each validator enforces a specific architectural constraint to ensure proper
separation of concerns in the ONEX 4-node architecture.

Related:
    - Ticket: OMN-1099 (Architecture Validator - Protocol Compliance)
    - PR: #124 (Protocol-Compliant Rule Classes)

Architecture Rules:
    - ARCH-001: No Direct Handler Dispatch
        Handlers must be dispatched through the runtime, not called directly.
        Direct calls bypass event tracking, circuit breaking, and other
        cross-cutting concerns.

    - ARCH-002: No Handler Publishing Events
        Handlers must not have direct event bus access. Only orchestrators
        may publish events. Handlers should return events for orchestrators
        to publish.

    - ARCH-003: No Workflow FSM in Orchestrators
        Orchestrators must not implement workflow FSMs (finite state machines).
        Reducers own state machines; orchestrators are "reaction planners"
        that coordinate work based on reducer outputs.

Two Interfaces:
    **1. Function-based validators** - Direct file validation, returns detailed results.

        Suitable for: Scripts, CLI tools, direct validation of single files.

        Example::

            from omnibase_infra.nodes.architecture_validator.validators import (
                validate_no_direct_dispatch,
            )

            result = validate_no_direct_dispatch("/path/to/handler.py")
            if not result.valid:
                for violation in result.violations:
                    print(f"{violation.location}: {violation.message}")

    **2. Protocol-compliant rule classes** - Implement `ProtocolArchitectureRule`.

        Suitable for: Integration with NodeArchitectureValidatorCompute, batch
        validation, registry-based rule management.

        Example::

            from omnibase_infra.nodes.architecture_validator.validators import (
                RuleNoDirectDispatch,
                RuleNoHandlerPublishing,
                RuleNoOrchestratorFSM,
            )

            rules = [
                RuleNoDirectDispatch(),
                RuleNoHandlerPublishing(),
                RuleNoOrchestratorFSM(),
            ]

            for rule in rules:
                result = rule.check("/path/to/file.py")
                if not result.passed:
                    print(f"{rule.rule_id}: {result.message}")

Thread Safety:
    All rule classes are stateless and safe for concurrent use across threads.
    Function-based validators are also thread-safe as they create new AST
    visitor instances for each invocation.

Configuration:
    Validators are wired through contract.yaml using the detection_strategy
    patterns. See the architecture validator node contract for configuration
    options and severity mappings.
"""

from __future__ import annotations

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

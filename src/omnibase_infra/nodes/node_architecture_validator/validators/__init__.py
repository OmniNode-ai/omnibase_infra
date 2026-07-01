# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validators for Architecture Validator node.

Validator implementations for ONEX architecture rules.
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

    - ARCH-004: Contract-Declared Orchestrator Workflow Must Be Bound To An Executor
        (Signal A.) Cross-file, node-directory rule (OMN-13472). An
        orchestrator-like node that declares an fsm:/workflow-state set must bind
        it to a runtime executor; it must not leave the contract table
        decorative while a handler drives the transitions itself. Catches the
        delegation-shaped anti-pattern (_transition(...) in a non-"*Orchestrator"
        handler) that ARCH-003's single-file, class-name-gated AST approach
        structurally misses. Reducers are exempt.

    - ARCH-004B: Orchestrator Handler Must Not Be A Class-B Monolith
        (Signal B.) DISTINCT rule from ARCH-004 Signal A (OMN-13486). Catches the
        orchestration-monolith complexity anti-pattern — a single oversized +
        complex orchestrator handler (high branch/event density or routing
        fan-in/out) that is not executor-bound — INDEPENDENT of FSM
        decorativeness. A node can fail Signal B with no FSM at all (build_loop /
        session / memory_lifecycle / swarm_dispatch). Owns its own B* finding
        codes and its own ratchet baseline dimension
        (architecture-handshakes/orchestration-monolith-baseline.yaml). Reducers
        are exempt.

Two Interfaces:
    **1. Function-based validators** - Direct file validation, returns detailed results.

        Suitable for: Scripts, CLI tools, direct validation of single files.

        Example::

            from omnibase_infra.nodes.node_architecture_validator.validators import (
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

            from omnibase_infra.nodes.node_architecture_validator.validators import (
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

from omnibase_infra.nodes.node_architecture_validator.validators.validator_contract_declared_orchestrator_workflow import (
    RuleContractDeclaredOrchestratorWorkflow,
    analyze_node_directory,
    validate_contract_declared_orchestrator_workflow,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_no_direct_dispatch import (
    RuleNoDirectDispatch,
    validate_no_direct_dispatch,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_no_handler_publishing import (
    RuleNoHandlerPublishing,
    validate_no_handler_publishing,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_no_orchestrator_fsm import (
    RuleNoOrchestratorFSM,
    validate_no_orchestrator_fsm,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_orchestration_monolith import (
    RuleOrchestrationMonolith,
    analyze_monolith,
    validate_orchestration_monolith,
)

__all__: list[str] = [
    # Functions (file-based validators)
    "validate_no_direct_dispatch",
    "validate_no_handler_publishing",
    "validate_no_orchestrator_fsm",
    # Functions (node-directory cross-file validators)
    "validate_contract_declared_orchestrator_workflow",
    "analyze_node_directory",
    # ARCH-004 Signal B (orchestration-monolith complexity, OMN-13486)
    "validate_orchestration_monolith",
    "analyze_monolith",
    # Classes (protocol-compliant rules)
    "RuleNoDirectDispatch",
    "RuleNoHandlerPublishing",
    "RuleNoOrchestratorFSM",
    "RuleContractDeclaredOrchestratorWorkflow",
    "RuleOrchestrationMonolith",
]

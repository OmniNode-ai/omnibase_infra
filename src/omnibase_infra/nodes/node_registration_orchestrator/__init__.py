# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Node package.

This node orchestrates the registration workflow by coordinating between
the reducer (for intent generation) and effect node (for execution).

Node Type: ORCHESTRATOR
Purpose: Coordinate node lifecycle registration workflows by consuming
         introspection events, requesting intents from reducer, and
         dispatching execution to the effect node.

The orchestrator extends NodeOrchestrator from omnibase_core, which provides:
- Workflow execution from YAML contracts
- Step dependency resolution
- Parallel/sequential execution modes
- Action emission for deferred execution

Exports:
    NodeRegistrationOrchestrator: Main orchestrator node implementation (declarative)
    ModelOrchestratorConfig: Configuration model
    ModelOrchestratorInput: Input model
    ModelOrchestratorOutput: Output model
    ModelIntentExecutionResult: Result model for intent execution
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registration_orchestrator.models import (
    ModelIntentExecutionResult,
    ModelOrchestratorConfig,
    ModelOrchestratorInput,
    ModelOrchestratorOutput,
)
from omnibase_infra.nodes.node_registration_orchestrator.node import (
    NodeRegistrationOrchestrator,
)

__all__: list[str] = [
    # Primary export - the declarative orchestrator
    "NodeRegistrationOrchestrator",
    # Models
    "ModelOrchestratorConfig",
    "ModelOrchestratorInput",
    "ModelOrchestratorOutput",
    "ModelIntentExecutionResult",
]

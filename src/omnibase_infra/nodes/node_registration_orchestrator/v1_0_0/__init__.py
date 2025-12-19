# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Node v1.0.0 package.

This version provides a declarative implementation of the Registration Orchestrator
Node that coordinates the node registration workflow using contract.yaml configuration.

The orchestrator extends NodeOrchestrator from omnibase_core, which provides:
- Workflow execution from YAML contracts
- Step dependency resolution
- Parallel/sequential execution modes
- Action emission for deferred execution

Exports:
    NodeRegistrationOrchestrator: Main orchestrator node implementation (declarative)

Note:
    Legacy models (ModelOrchestratorConfig, ModelOrchestratorInput, ModelOrchestratorOutput,
    ModelIntentExecutionResult) are kept in the models/ directory for backwards compatibility
    with existing tests and integrations. New code should use omnibase_core models instead:
    - omnibase_core.models.orchestrator.ModelOrchestratorInput
    - omnibase_core.models.orchestrator.ModelOrchestratorOutput
"""

from __future__ import annotations

# Re-export legacy models for backwards compatibility
# These will be deprecated in favor of omnibase_core models
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models import (
    ModelIntentExecutionResult,
    ModelOrchestratorConfig,
    ModelOrchestratorInput,
    ModelOrchestratorOutput,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.node import (
    NodeRegistrationOrchestrator,
)

__all__: list[str] = [
    # Primary export - the declarative orchestrator
    "NodeRegistrationOrchestrator",
    # Legacy models (kept for backwards compatibility)
    "ModelOrchestratorConfig",
    "ModelOrchestratorInput",
    "ModelOrchestratorOutput",
    "ModelIntentExecutionResult",
]

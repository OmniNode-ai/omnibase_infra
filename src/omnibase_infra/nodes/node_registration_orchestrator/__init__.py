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

Event Handlers (all co-located in handlers/ subdirectory):
    - HandlerNodeIntrospected: Processes NodeIntrospectionEvent (canonical trigger)
    - HandlerNodeRegistrationAcked: Processes NodeRegistrationAcked commands
    - HandlerRuntimeTick: Processes RuntimeTick for timeout evaluation
    - HandlerNodeHeartbeat: Processes NodeHeartbeat for liveness tracking (OMN-1006)

    For handler access, import from handlers submodule:
    ```python
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
        ModelHeartbeatHandlerResult,
    )
    ```

Exports:
    NodeRegistrationOrchestrator: Main orchestrator node implementation (declarative)
    TimeoutCoordinator: Coordinator for RuntimeTick timeout coordination
    ModelTimeoutCoordinationResult: Result model for timeout coordinator
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
from omnibase_infra.nodes.node_registration_orchestrator.timeout_coordinator import (
    ModelTimeoutCoordinationResult,
    TimeoutCoordinator,
)

__all__: list[str] = [
    # Primary export - the declarative orchestrator
    "NodeRegistrationOrchestrator",
    # Coordinators
    "TimeoutCoordinator",
    "ModelTimeoutCoordinationResult",
    # Models
    "ModelOrchestratorConfig",
    "ModelOrchestratorInput",
    "ModelOrchestratorOutput",
    "ModelIntentExecutionResult",
]

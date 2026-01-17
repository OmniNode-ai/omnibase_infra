# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Node package.

This node orchestrates the registration workflow by coordinating between
the reducer (for intent generation) and effect node (for execution).

Node Type: ORCHESTRATOR_GENERIC
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

# Dispatchers (moved from runtime/dispatchers/ - OMN-1346)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
    DispatcherNodeIntrospected,
    DispatcherNodeRegistrationAcked,
    DispatcherRuntimeTick,
)
from omnibase_infra.nodes.node_registration_orchestrator.introspection_event_router import (
    IntrospectionEventRouter,
)
from omnibase_infra.nodes.node_registration_orchestrator.models import (
    ModelIntentExecutionResult,
    ModelOrchestratorConfig,
    ModelOrchestratorInput,
    ModelOrchestratorOutput,
)
from omnibase_infra.nodes.node_registration_orchestrator.node import (
    NodeRegistrationOrchestrator,
)

# Domain plugin (OMN-1346) - kernel initialization plugin
from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
    PluginRegistration,
)
from omnibase_infra.nodes.node_registration_orchestrator.timeout_coordinator import (
    ModelTimeoutCoordinationResult,
    TimeoutCoordinator,
)

# Domain wiring (OMN-1346) - handler/dispatcher wiring, getters, and route ID constants
from omnibase_infra.nodes.node_registration_orchestrator.wiring import (
    ROUTE_ID_NODE_INTROSPECTION,
    ROUTE_ID_NODE_REGISTRATION_ACKED,
    ROUTE_ID_RUNTIME_TICK,
    get_handler_node_introspected_from_container,
    get_handler_node_registration_acked_from_container,
    get_handler_runtime_tick_from_container,
    get_projection_reader_from_container,
    wire_registration_dispatchers,
    wire_registration_handlers,
)

__all__: list[str] = [
    # Dispatchers (moved from runtime/dispatchers/ - OMN-1346)
    "DispatcherNodeIntrospected",
    "DispatcherNodeRegistrationAcked",
    "DispatcherRuntimeTick",
    # Event routing (moved from runtime/ - OMN-1346)
    "IntrospectionEventRouter",
    "ModelIntentExecutionResult",
    # Models
    "ModelOrchestratorConfig",
    "ModelOrchestratorInput",
    "ModelOrchestratorOutput",
    "ModelTimeoutCoordinationResult",
    # Primary export - the declarative orchestrator
    "NodeRegistrationOrchestrator",
    # Domain plugin (OMN-1346)
    "PluginRegistration",
    # Route ID constants (OMN-1346)
    "ROUTE_ID_NODE_INTROSPECTION",
    "ROUTE_ID_NODE_REGISTRATION_ACKED",
    "ROUTE_ID_RUNTIME_TICK",
    # Coordinators
    "TimeoutCoordinator",
    # Domain wiring - handler and dispatcher wiring (OMN-1346)
    "wire_registration_dispatchers",
    "wire_registration_handlers",
    # Domain wiring - handler getters (OMN-1346)
    "get_projection_reader_from_container",
    "get_handler_node_introspected_from_container",
    "get_handler_runtime_tick_from_container",
    "get_handler_node_registration_acked_from_container",
]

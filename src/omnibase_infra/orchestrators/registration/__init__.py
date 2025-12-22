# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Node (C1).

This module exports the NodeRegistrationOrchestrator which coordinates the
node registration workflow following ONEX architectural constraints.

The orchestrator:
    - Consumes NodeIntrospected events (canonical trigger)
    - Consumes NodeRegistrationAcked commands (gated flows)
    - Consumes RuntimeTick for timeout evaluation
    - Emits decision events (NodeRegistrationInitiated, NodeRegistrationAccepted, etc.)
    - Uses ProjectionReaderRegistration for state queries
    - Uses injected `now` for deadline calculations

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator (Event-Driven)
    - OMN-930 (C0): Projection Reader Interface
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-949 (B6): Runtime Scheduler / RuntimeTick

Exports:
    NodeRegistrationOrchestrator: The registration workflow orchestrator node
    ModelOrchestratorContext: Context model for orchestrator handlers
    HandlerNodeIntrospected: Handler for NodeIntrospectionEvent
    HandlerNodeRegistrationAcked: Handler for NodeRegistrationAcked command
    HandlerRuntimeTick: Handler for RuntimeTick timeout detection
"""

from omnibase_infra.orchestrators.registration.handlers import (
    HandlerNodeIntrospected,
    HandlerNodeRegistrationAcked,
    HandlerRuntimeTick,
)
from omnibase_infra.orchestrators.registration.models.model_orchestrator_context import (
    ModelOrchestratorContext,
)
from omnibase_infra.orchestrators.registration.node_registration_orchestrator import (
    NodeRegistrationOrchestrator,
)

__all__: list[str] = [
    "HandlerNodeIntrospected",
    "HandlerNodeRegistrationAcked",
    "HandlerRuntimeTick",
    "ModelOrchestratorContext",
    "NodeRegistrationOrchestrator",
]

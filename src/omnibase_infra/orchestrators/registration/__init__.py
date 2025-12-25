# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Node (C1).

This module exports the NodeRegistrationOrchestrator which coordinates the
node registration workflow following ONEX architectural constraints.

The orchestrator:
    - Consumes NodeIntrospected events (canonical trigger)
    - Consumes NodeRegistrationAcked commands (gated flows)
    - Consumes RuntimeTick for timeout evaluation
    - Consumes NodeHeartbeat events for liveness tracking
    - Emits decision events (NodeRegistrationInitiated, NodeRegistrationAccepted, etc.)
    - Uses ProjectionReaderRegistration for state queries
    - Uses injected `now` for deadline calculations

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator (Event-Driven)
    - OMN-930 (C0): Projection Reader Interface
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-949 (B6): Runtime Scheduler / RuntimeTick
    - OMN-1006: Node Heartbeat for Liveness Tracking

Exports:
    NodeRegistrationOrchestrator: The registration workflow orchestrator node
    HandlerNodeIntrospected: Handler for NodeIntrospectionEvent
    HandlerNodeRegistrationAcked: Handler for NodeRegistrationAcked command
    HandlerRuntimeTick: Handler for RuntimeTick timeout detection
    HandlerNodeHeartbeat: Handler for node heartbeat events
    DEFAULT_LIVENESS_WINDOW_SECONDS: Default liveness window (90 seconds)
    ModelHeartbeatHandlerResult: Result model for heartbeat processing

Note:
    For orchestrator context, use the canonical ModelOrchestratorContext from
    omnibase_core.models.orchestrator, which provides time injection and
    correlation tracking for orchestrator handler execution.
"""

from omnibase_infra.orchestrators.registration.handlers import (
    DEFAULT_LIVENESS_WINDOW_SECONDS,
    HandlerNodeHeartbeat,
    HandlerNodeIntrospected,
    HandlerNodeRegistrationAcked,
    HandlerRuntimeTick,
    ModelHeartbeatHandlerResult,
)
from omnibase_infra.orchestrators.registration.node_registration_orchestrator import (
    NodeRegistrationOrchestrator,
)

__all__: list[str] = [
    "DEFAULT_LIVENESS_WINDOW_SECONDS",
    "HandlerNodeHeartbeat",
    "HandlerNodeIntrospected",
    "HandlerNodeRegistrationAcked",
    "HandlerRuntimeTick",
    "ModelHeartbeatHandlerResult",
    "NodeRegistrationOrchestrator",
]

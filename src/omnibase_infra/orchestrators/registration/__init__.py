# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Handlers.

This module exports the registration workflow handlers following ONEX
architectural constraints. The handlers are used by the declarative
NodeRegistrationOrchestrator located at:
    nodes/node_registration_orchestrator/node.py

Handler Responsibilities:
    - HandlerNodeIntrospected: Process NodeIntrospected events (canonical trigger)
    - HandlerNodeRegistrationAcked: Process NodeRegistrationAcked commands (gated flows)
    - HandlerRuntimeTick: Process RuntimeTick for timeout evaluation
    - HandlerNodeHeartbeat: Process NodeHeartbeat events for liveness tracking

All handlers:
    - Use ProjectionReaderRegistration for state queries
    - Use injected `now` for deadline calculations
    - Emit EVENTS only (no intents, no projections)

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator (Event-Driven)
    - OMN-930 (C0): Projection Reader Interface
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-949 (B6): Runtime Scheduler / RuntimeTick
    - OMN-1006: Node Heartbeat for Liveness Tracking

Exports:
    HandlerNodeIntrospected: Handler for NodeIntrospectionEvent
    HandlerNodeRegistrationAcked: Handler for NodeRegistrationAcked command
    HandlerRuntimeTick: Handler for RuntimeTick timeout detection
    HandlerNodeHeartbeat: Handler for node heartbeat events
    DEFAULT_LIVENESS_WINDOW_SECONDS: Default liveness window (90 seconds)
    ModelHeartbeatHandlerResult: Result model for heartbeat processing

Note:
    For the declarative orchestrator, import from:
        omnibase_infra.nodes.node_registration_orchestrator
"""

from omnibase_infra.orchestrators.registration.handlers import (
    DEFAULT_LIVENESS_WINDOW_SECONDS,
    HandlerNodeHeartbeat,
    HandlerNodeIntrospected,
    HandlerNodeRegistrationAcked,
    HandlerRuntimeTick,
    ModelHeartbeatHandlerResult,
)

__all__: list[str] = [
    "DEFAULT_LIVENESS_WINDOW_SECONDS",
    "HandlerNodeHeartbeat",
    "HandlerNodeIntrospected",
    "HandlerNodeRegistrationAcked",
    "HandlerRuntimeTick",
    "ModelHeartbeatHandlerResult",
]

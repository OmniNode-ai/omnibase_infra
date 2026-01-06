# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Handlers.

This module exports handler implementations for the NodeRegistrationOrchestrator.
Each handler processes a specific event type and returns events only.

Handlers:
    - HandlerNodeIntrospected: Processes NodeIntrospectionEvent (canonical trigger)
    - HandlerNodeRegistrationAcked: Processes NodeRegistrationAcked commands
    - HandlerRuntimeTick: Processes RuntimeTick for timeout evaluation
    - HandlerNodeHeartbeat: Processes NodeHeartbeat for liveness tracking

All handlers follow the pattern:
    async def handle(event, now, correlation_id) -> list[BaseModel]

Handler Architecture:
    - Handlers are stateless classes (no mutable state between calls)
    - Handlers use projection reader for state queries
    - Handlers may use projector for state persistence (e.g., HandlerNodeIntrospected)
    - Handlers use `now` parameter for time-based decisions
    - Handlers return EVENTS only (never intents or projections)

Projection Persistence:
    Handlers that modify registration state can optionally accept a projector for
    persisting projections to PostgreSQL. When configured with a projector, the
    handler persists the state transition BEFORE returning events, ensuring that
    read models are consistent before downstream processing.

    Example:
        handler = HandlerNodeIntrospected(reader, projector=projector)
        events = await handler.handle(event, now, correlation_id)
        # Projection was persisted before events were returned

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-1006: Node Heartbeat for Liveness Tracking
    - OMN-892: 2-Way Registration E2E Integration Test
"""

from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_heartbeat import (
    DEFAULT_LIVENESS_WINDOW_SECONDS,
    HandlerNodeHeartbeat,
    ModelHeartbeatHandlerResult,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_registration_acked import (
    DEFAULT_LIVENESS_INTERVAL_SECONDS,
    ENV_LIVENESS_INTERVAL_SECONDS,
    HandlerNodeRegistrationAcked,
    get_liveness_interval_seconds,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick import (
    HandlerRuntimeTick,
)

__all__: list[str] = [
    "DEFAULT_LIVENESS_INTERVAL_SECONDS",
    "DEFAULT_LIVENESS_WINDOW_SECONDS",
    "ENV_LIVENESS_INTERVAL_SECONDS",
    "HandlerNodeHeartbeat",
    "HandlerNodeIntrospected",
    "HandlerNodeRegistrationAcked",
    "HandlerRuntimeTick",
    "ModelHeartbeatHandlerResult",
    "get_liveness_interval_seconds",
]

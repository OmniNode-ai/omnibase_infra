# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Handlers.

This module exports handler implementations for the NodeRegistrationOrchestrator.
Each handler processes a specific event type and returns events only.

Handlers:
    - HandlerNodeIntrospected: Processes NodeIntrospectionEvent (canonical trigger)
    - HandlerNodeRegistrationAcked: Processes NodeRegistrationAcked commands
    - HandlerRuntimeTick: Processes RuntimeTick for timeout evaluation

All handlers follow the pattern:
    async def handle(event, now, correlation_id) -> list[BaseModel]

Handler Architecture:
    - Handlers are stateless classes (no mutable state between calls)
    - Handlers use projection reader for state queries (read-only)
    - Handlers use `now` parameter for time-based decisions
    - Handlers return EVENTS only (never intents or projections)

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
"""

from omnibase_infra.orchestrators.registration.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.orchestrators.registration.handlers.handler_node_registration_acked import (
    HandlerNodeRegistrationAcked,
)
from omnibase_infra.orchestrators.registration.handlers.handler_runtime_tick import (
    HandlerRuntimeTick,
)

__all__: list[str] = [
    "HandlerNodeIntrospected",
    "HandlerNodeRegistrationAcked",
    "HandlerRuntimeTick",
]

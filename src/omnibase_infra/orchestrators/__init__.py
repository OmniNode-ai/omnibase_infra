# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Orchestrator Handlers.

This module exports orchestrator handler implementations for the ONEX infrastructure.
Handlers are used by declarative orchestrator nodes defined in the nodes/ directory.

Key Architectural Constraints (from ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md):
    - Orchestrators own workflow and time
    - Orchestrators emit EVENTS only
    - Orchestrators receive `now: datetime` injected by runtime
    - Orchestrators NEVER perform I/O
    - Orchestrators read state from projections (via ProjectionReader)

For declarative orchestrators, import from:
    omnibase_infra.nodes.node_registration_orchestrator

Exports:
    DEFAULT_LIVENESS_WINDOW_SECONDS: Default liveness window (90 seconds)
    HandlerNodeHeartbeat: Handler for node heartbeat events
    ModelHeartbeatHandlerResult: Result model for heartbeat processing
"""

from omnibase_infra.orchestrators.registration import (
    DEFAULT_LIVENESS_WINDOW_SECONDS,
    HandlerNodeHeartbeat,
    ModelHeartbeatHandlerResult,
)

__all__: list[str] = [
    "DEFAULT_LIVENESS_WINDOW_SECONDS",
    "HandlerNodeHeartbeat",
    "ModelHeartbeatHandlerResult",
]

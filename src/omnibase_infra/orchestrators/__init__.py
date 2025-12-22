# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Orchestrator Nodes.

This module exports orchestrator node implementations for the ONEX infrastructure.
Orchestrators coordinate workflows, make time-dependent decisions, and emit EVENTS
only (never intents or projections).

Key Architectural Constraints (from ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md):
    - Orchestrators own workflow and time
    - Orchestrators emit EVENTS only
    - Orchestrators receive `now: datetime` injected by runtime
    - Orchestrators NEVER perform I/O
    - Orchestrators read state from projections (via ProjectionReader)

Exports:
    NodeRegistrationOrchestrator: Registration workflow orchestrator (C1)
"""

from omnibase_infra.orchestrators.registration import NodeRegistrationOrchestrator

__all__: list[str] = [
    "NodeRegistrationOrchestrator",
]

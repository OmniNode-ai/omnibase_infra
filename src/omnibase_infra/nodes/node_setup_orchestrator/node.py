# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative ORCHESTRATOR node for platform setup coordination.

Coordinates the full setup workflow: cloud gate check → preflight validation
→ local service provisioning → Infisical bootstrap → post-provision validation.

All effect node dependencies are injected via protocol interfaces, enabling
full test isolation without subprocess or Docker dependencies.

Invariants:
    I5 — Orchestrator output has no ``result`` field.
    I6 — All emitted event types are in SETUP_EVENT_TYPES frozenset.
    I8 — Cloud gate is a hard stop before preflight.

Ticket: OMN-3495
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeSetupOrchestrator(NodeOrchestrator):
    """Orchestrates the full platform setup workflow.

    Coordinates 4 effect nodes in sequence:
        1. Cloud gate check (I8 hard stop)
        2. Preflight validation (NodePreflightEffect)
        3. Local service provisioning (NodeLocalProvisionEffect)
        4. Infisical bootstrap (NodeInfisicalSetupEffect, if enabled)
        5. Post-provision validation (NodeSetupValidateEffect)

    The active handler (HandlerSetupOrchestrator) is injected via the
    registry and protocol interfaces — this class only provides the
    declarative node identity.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the setup orchestrator node."""
        super().__init__(container)


__all__: list[str] = ["NodeSetupOrchestrator"]

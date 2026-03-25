# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative ORCHESTRATOR node for runtime boot sequence [OMN-6351].

Coordinates the 4-step boot sequence:
    1. contract_loader (Effect) -- scan filesystem for contracts
    2. contract_registry (Reducer) -- project contracts to registry
    3. node_graph (Reducer) -- instantiate node graph from contracts
    4. event_bus_wiring (Effect) -- wire nodes to Kafka subscriptions

Sequential execution with fail-fast semantics. Step dependencies are
resolved via container.get_service().

Ticket: OMN-6351
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeRuntimeOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for the ONEX runtime boot sequence.

    Handlers:
        - ``HandlerRuntimeLifecycle``: Coordinates 4-step sequential boot.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the runtime orchestrator node."""
        super().__init__(container)

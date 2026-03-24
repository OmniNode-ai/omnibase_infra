# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative EFFECT node wrapping EventBusSubcontractWiring [OMN-6350].

Wires instantiated nodes to event bus subscriptions as part of the ONEX
runtime self-hosting architecture. All behavior is defined in contract.yaml
and delegated to handlers.

Ticket: OMN-6350
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeEventBusWiringEffect(NodeEffect):
    """Declarative effect node for event bus wiring.

    Handlers:
        - ``HandlerEventBusWiring``: Wires nodes to Kafka subscriptions.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the event bus wiring effect node."""
        super().__init__(container)

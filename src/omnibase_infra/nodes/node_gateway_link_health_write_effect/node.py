# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeGatewayLinkHealthWriteEffect - declarative EFFECT node.

Persists ModelPayloadGatewayLinkHealthUpsert intents into the
omninode_internal.gateway_link_health latest-known-state table. All persistence logic
lives in HandlerGatewayLinkHealthUpsert per the ONEX declarative pattern.

Ticket: OMN-15570 (G3, gateway lift Phase 0)
"""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeGatewayLinkHealthWriteEffect(NodeEffect):
    """Declarative EFFECT node for gateway_link_health persistence.

    All behavior is defined in contract.yaml and delegated to
    HandlerGatewayLinkHealthUpsert. This node contains no custom logic
    beyond the explicit DI constructor required by the nodes/*/node.py
    guideline.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeGatewayLinkHealthWriteEffect"]

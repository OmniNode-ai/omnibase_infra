# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeGatewayLinkHealthProjectionCompute - declarative COMPUTE node.

Subscribes to the gateway heartbeat event topic and delegates all compute
logic to HandlerGatewayLinkHealthProjection per the ONEX declarative pattern.

Subscribed Topic (via contract.yaml):
    - onex.evt.omnibase-infra.gateway-heartbeat.v1

Ticket: OMN-15570 (G3, gateway lift Phase 0)
"""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_compute import NodeCompute


class NodeGatewayLinkHealthProjectionCompute(NodeCompute):
    """Declarative COMPUTE node for gateway link-health projection.

    All behavior is defined in contract.yaml and delegated to
    HandlerGatewayLinkHealthProjection. This node contains no custom logic
    beyond the explicit DI constructor required by the nodes/*/node.py
    guideline.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeGatewayLinkHealthProjectionCompute"]

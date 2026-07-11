# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodePrStateProjectionCompute - declarative COMPUTE node.

Subscribes to the GitHub PR status event topic and delegates all compute
logic to HandlerPrStateProjection per the ONEX declarative pattern.

Subscribed Topic (via contract.yaml):
    - onex.evt.github.pr-status.v1

Ticket: OMN-14375
"""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_compute import NodeCompute


class NodePrStateProjectionCompute(NodeCompute):
    """Declarative COMPUTE node for GitHub PR status projection.

    All behavior is defined in contract.yaml and delegated to
    HandlerPrStateProjection. This node contains no custom logic beyond the
    explicit DI constructor required by the nodes/*/node.py guideline.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodePrStateProjectionCompute"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeBuildLoopProjectionCompute - declarative COMPUTE node.

Subscribes to the build_loop_workflow terminal event topic and delegates all
compute logic to HandlerBuildLoopProjection per the ONEX declarative pattern.

Subscribed Topic (via contract.yaml):
    - onex.evt.omnimarket.build-loop-orchestrator-completed.v1

Ticket: OMN-9774
"""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_compute import NodeCompute


class NodeBuildLoopProjectionCompute(NodeCompute):
    """Declarative COMPUTE node for build_loop terminal-event projection.

    All behavior is defined in contract.yaml and delegated to
    HandlerBuildLoopProjection. This node contains no custom logic beyond the
    explicit DI constructor required by the nodes/*/node.py guideline.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeBuildLoopProjectionCompute"]

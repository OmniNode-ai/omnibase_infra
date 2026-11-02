# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node for the wait-while-busy deploy test (OMN-21866)."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeLabLaneComposeBusyCompute(NodeCompute):
    """Decide whether an answering lab lane's deploy agent reads a compose deploy."""

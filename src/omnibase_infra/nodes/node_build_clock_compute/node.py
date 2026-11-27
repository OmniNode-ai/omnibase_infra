# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node: container start to first healthy /health."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeBuildClockCompute(NodeCompute):
    """Compute the build clock and verdict; the handler owns the logic."""

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node; the handler owns the priority decision."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeQueuedRequestPriorityCompute(NodeCompute):
    """Decide which queued rebuild request runs for a window."""

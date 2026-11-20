# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node; the handler owns the window decision."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeDevRecreateWindowCompute(NodeCompute):
    """Decide whether the dev runtime may be recreated now."""

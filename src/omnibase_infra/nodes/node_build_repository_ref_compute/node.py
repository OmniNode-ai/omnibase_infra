# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node; the handler owns the verdict."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeBuildRepositoryRefCompute(NodeCompute):
    """Decide whether the build repository holds the commit a queued build names."""

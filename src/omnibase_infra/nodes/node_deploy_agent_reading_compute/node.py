# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node; the handler owns the judgment."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeDeployAgentReadingCompute(NodeCompute):
    """Judge the dev deploy agent ticket's restart, build and untouched-agent readings."""

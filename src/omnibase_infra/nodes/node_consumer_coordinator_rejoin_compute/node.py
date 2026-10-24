# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node for coordinator recovery (OMN-21337)."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeConsumerCoordinatorRejoinCompute(NodeCompute):
    """Decide coordinator recovery through the contract's pure handler."""

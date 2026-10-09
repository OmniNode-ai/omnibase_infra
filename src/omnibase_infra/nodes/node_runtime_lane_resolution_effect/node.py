# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative node for startup lane overlay resolution (OMN-19747)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeLaneResolutionEffect(NodeEffect):
    """Read the deployment lane overlay through HandlerRuntimeLaneResolution."""

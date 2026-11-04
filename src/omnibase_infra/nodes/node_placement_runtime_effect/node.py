# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative node; placement logic belongs to the routed handler."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodePlacementRuntimeEffect(NodeEffect):
    """Place one runtime's loaded nodes by their placement policy."""

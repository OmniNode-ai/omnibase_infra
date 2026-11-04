# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host-local image maintenance node (OMN-19998)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDeployAgentImagePruneEffect(NodeEffect):
    """The contract routes planning, pruning and installation to the handler."""

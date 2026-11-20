# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDeploySourceCloneEffect(NodeEffect):
    """Resolve and create the deploy agent's own deploy-source clone."""

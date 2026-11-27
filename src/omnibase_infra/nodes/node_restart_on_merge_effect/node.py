# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRestartOnMergeEffect(NodeEffect):
    """Decide whether a merge may sync and restart one deploy-agent unit."""

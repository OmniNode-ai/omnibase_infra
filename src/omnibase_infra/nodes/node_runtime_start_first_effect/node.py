# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative start-first replacement effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeStartFirstEffect(NodeEffect):
    """Replace one dev runtime container start-first."""

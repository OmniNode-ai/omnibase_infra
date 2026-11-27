# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative node for the lazy-start front door (OMN-21592)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeLazyStartIngress(NodeEffect):
    """Start a dormant contract on its first request and deliver that request."""

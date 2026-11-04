# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative runtime listener holder effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeListenerHolderEffect(NodeEffect):
    """Deploy-agent holder of the runtime host listeners (OMN-22379)."""

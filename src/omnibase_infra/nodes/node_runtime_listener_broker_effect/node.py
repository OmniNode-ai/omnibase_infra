# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative runtime listener broker effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeListenerBrokerEffect(NodeEffect):
    """Deploy-agent broker of the runtime host listeners (OMN-22379)."""

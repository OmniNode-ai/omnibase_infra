# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host boot effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDevBootGuardEffect(NodeEffect):
    """Supervise the host's dev runtime boot and retained revision."""

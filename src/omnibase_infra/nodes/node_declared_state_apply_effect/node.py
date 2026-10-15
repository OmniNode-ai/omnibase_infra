# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect node for declared-state apply."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDeclaredStateApplyEffect(NodeEffect):
    """Apply a planned diff through the contract's handler."""

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative standalone-service recreate effect; the handler owns the operation."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeStandaloneComposeRecreateEffect(NodeEffect):
    """Recreate one declared service of one declared standalone compose file."""

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative slot teardown evidence effect."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodePreprTeardownSlot(NodeEffect):
    """Tenant offboard and report evidence owned by the bound handler."""

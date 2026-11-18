# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative C11 delegation-legs probe effect node (OMN-23177)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeC11DelegationLegsEffect(NodeEffect):
    """All probe behaviour is owned by the contract-routed handler."""

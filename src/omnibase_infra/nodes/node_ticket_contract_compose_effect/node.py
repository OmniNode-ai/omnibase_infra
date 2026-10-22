# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative node; filesystem composition belongs to the routed handler."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeTicketContractComposeEffect(NodeEffect):
    """Read branch fragments and emit a composed ticket contract."""

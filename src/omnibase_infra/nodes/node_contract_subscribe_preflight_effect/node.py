# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative pre-subscribe effect; the handler owns the operation (OMN-21753)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeContractSubscribePreflightEffect(NodeEffect):
    """Prepare a contract's consumer on the bus before its subscribe joins."""

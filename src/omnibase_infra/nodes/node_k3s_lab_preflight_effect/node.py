# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns the readings."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeK3sLabPreflightEffect(NodeEffect):
    """Refuse a lab k3s server start whose pinned node-ip no live interface holds."""

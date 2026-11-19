# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns the read."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeVerifyStabilityRefresh(NodeEffect):
    """Read the stability broker's Kafka API as the refresh health gate's principal."""

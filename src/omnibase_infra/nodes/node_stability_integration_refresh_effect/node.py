# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns the refresh."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeStabilityIntegrationRefreshEffect(NodeEffect):
    """Build the integration branch heads and run them on the stability lane."""

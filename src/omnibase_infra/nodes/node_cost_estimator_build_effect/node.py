# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host build effect; all operations belong to its handler."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeCostEstimatorBuildEffect(NodeEffect):
    """Build the estimator image before the deploy agent recreates the page."""

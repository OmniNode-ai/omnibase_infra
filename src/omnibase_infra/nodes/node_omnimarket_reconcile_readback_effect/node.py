# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative node for the post-reconcile provenance readback."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeOmnimarketReconcileReadbackEffect(NodeEffect):
    """Read installed and canonical identities after reconciliation."""

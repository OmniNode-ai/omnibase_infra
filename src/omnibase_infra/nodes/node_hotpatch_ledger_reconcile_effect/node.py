# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative hot-patch ledger reconcile effect node (OMN-17427)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeHotpatchLedgerReconcileEffect(NodeEffect):
    """All reconcile behaviour is owned by the contract-routed handler."""

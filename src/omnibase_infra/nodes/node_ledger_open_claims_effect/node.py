# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns the reads."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeLedgerOpenClaimsEffect(NodeEffect):
    """Read the open CLAIMs of a window from the reader node, or refuse; never read the ledger file."""

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Delegation-chain ledger writer — declarative EFFECT node (OMN-16964)."""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeDelegationChainLedgerEffect(NodeEffect):
    """Persist re-derived delegation-chain evidence through its handler."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeDelegationChainLedgerEffect"]

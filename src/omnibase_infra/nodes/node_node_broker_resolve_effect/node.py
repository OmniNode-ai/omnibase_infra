# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect node for resolving a node's broker from its contract overlay (OMN-22362)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeNodeBrokerResolveEffect(NodeEffect):
    """Resolve a node's broker through the contract-bound handler."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the node broker resolution effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeNodeBrokerResolveEffect"]

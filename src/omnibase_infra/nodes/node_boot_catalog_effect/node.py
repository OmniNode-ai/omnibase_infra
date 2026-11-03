# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect node for boot step 0, the boot catalog (OMN-21590)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeBootCatalogEffect(NodeEffect):
    """Announce installed packages' contracts through the contract-bound handler."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the boot catalog effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeBootCatalogEffect"]

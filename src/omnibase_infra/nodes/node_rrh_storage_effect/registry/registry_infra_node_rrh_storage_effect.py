# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DI registry for the RRH storage effect node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_rrh_storage_effect.node import NodeRRHStorageEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class RegistryInfraNodeRRHStorageEffect:
    """DI registry for ``NodeRRHStorageEffect``."""

    @staticmethod
    def get_node_class() -> type[NodeRRHStorageEffect]:
        """Return the node class for DI resolution."""
        return NodeRRHStorageEffect

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodeRRHStorageEffect:
        """Create a node instance with the given container."""
        return NodeRRHStorageEffect(container)


__all__: list[str] = ["RegistryInfraNodeRRHStorageEffect"]

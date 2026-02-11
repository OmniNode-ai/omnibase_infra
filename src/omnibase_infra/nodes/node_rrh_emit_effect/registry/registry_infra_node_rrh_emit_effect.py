# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DI registry for the RRH emit effect node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_rrh_emit_effect.node import NodeRRHEmitEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class RegistryInfraNodeRRHEmitEffect:
    """DI registry for ``NodeRRHEmitEffect``."""

    @staticmethod
    def get_node_class() -> type[NodeRRHEmitEffect]:
        """Return the node class for DI resolution."""
        return NodeRRHEmitEffect

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodeRRHEmitEffect:
        """Create a node instance with the given container."""
        return NodeRRHEmitEffect(container)


__all__: list[str] = ["RegistryInfraNodeRRHEmitEffect"]

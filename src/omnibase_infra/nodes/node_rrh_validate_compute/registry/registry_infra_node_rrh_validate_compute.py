# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DI registry for the RRH validate compute node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_rrh_validate_compute.node import NodeRRHValidateCompute

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class RegistryInfraNodeRRHValidateCompute:
    """DI registry for ``NodeRRHValidateCompute``."""

    @staticmethod
    def get_node_class() -> type[NodeRRHValidateCompute]:
        """Return the node class for DI resolution."""
        return NodeRRHValidateCompute

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodeRRHValidateCompute:
        """Create a node instance with the given container."""
        return NodeRRHValidateCompute(container)


__all__: list[str] = ["RegistryInfraNodeRRHValidateCompute"]

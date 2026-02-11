# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeAuthGateCompute — DI bindings and exports.

Provides factory methods and dependency injection bindings for
the NodeAuthGateCompute. Follows the ONEX registry pattern with
the naming convention ``RegistryInfra<NodeName>``.

Ticket: OMN-2125
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_auth_gate_compute.node import (
    NodeAuthGateCompute,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraAuthGateCompute:
    """DI registry for auth gate compute node.

    Provides factory methods and bindings for the NodeAuthGateCompute.
    This registry follows the ONEX registry pattern for infrastructure nodes.
    """

    @staticmethod
    def get_node_class() -> type[NodeAuthGateCompute]:
        """Return the node class for DI resolution.

        Returns:
            The NodeAuthGateCompute class type.
        """
        return NodeAuthGateCompute

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodeAuthGateCompute:
        """Create a NodeAuthGateCompute instance with the given container.

        Args:
            container: ONEX dependency injection container.

        Returns:
            Configured NodeAuthGateCompute instance.
        """
        return NodeAuthGateCompute(container)


__all__: list[str] = [
    "NodeAuthGateCompute",
    "RegistryInfraAuthGateCompute",
]

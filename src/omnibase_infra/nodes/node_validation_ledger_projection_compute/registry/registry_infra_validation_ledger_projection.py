# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeValidationLedgerProjectionCompute - DI bindings and exports.

Follows the ONEX registry pattern with naming convention
``RegistryInfra<NodeName>``.

Ticket: OMN-1908
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_validation_ledger_projection_compute.node import (
    NodeValidationLedgerProjectionCompute,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraValidationLedgerProjection:
    """DI registry for validation ledger projection compute node.

    Provides factory methods and bindings for the
    NodeValidationLedgerProjectionCompute.
    """

    @staticmethod
    def get_node_class() -> type[NodeValidationLedgerProjectionCompute]:
        """Return the node class for DI resolution."""
        return NodeValidationLedgerProjectionCompute

    @staticmethod
    def create_node(
        container: ModelONEXContainer,
    ) -> NodeValidationLedgerProjectionCompute:
        """Create a NodeValidationLedgerProjectionCompute instance.

        Args:
            container: ONEX dependency injection container.

        Returns:
            Configured node instance.
        """
        return NodeValidationLedgerProjectionCompute(container)


__all__ = [
    "RegistryInfraValidationLedgerProjection",
]

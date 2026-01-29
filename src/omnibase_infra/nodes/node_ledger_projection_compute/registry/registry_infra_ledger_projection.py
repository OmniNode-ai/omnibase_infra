# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeLedgerProjectionCompute - DI bindings and exports.

This registry provides factory methods and dependency injection bindings for
the NodeLedgerProjectionCompute. It follows the ONEX registry pattern with
the naming convention ``RegistryInfra<NodeName>``.

Node Purpose:
    NodeLedgerProjectionCompute is a COMPUTE node that transforms platform
    events (ModelEventMessage) into ledger append intents (ModelPayloadLedgerAppend)
    for persistence by the Effect layer.

Usage:
    ```python
    from omnibase_infra.nodes.node_ledger_projection_compute.registry import (
        RegistryInfraLedgerProjection,
    )

    # Get the node class for DI resolution
    node_class = RegistryInfraLedgerProjection.get_node_class()

    # Create node instance with container
    from omnibase_core.container import ModelONEXContainer
    container = ModelONEXContainer()
    node = RegistryInfraLedgerProjection.create_node(container)
    ```

Related Tickets:
    - OMN-1648: Ledger Projection Compute Node
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_ledger_projection_compute.node import (
    NodeLedgerProjectionCompute,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraLedgerProjection:
    """DI registry for ledger projection compute node.

    Provides factory methods and bindings for the NodeLedgerProjectionCompute.
    This registry follows the ONEX registry pattern for infrastructure nodes.

    Why a Registry Class?
        ONEX convention requires registry classes (not modules) for:
        - Centralized factory methods for node creation
        - Type-safe DI resolution via get_node_class()
        - Extensibility for subclassing specialized registries
        - Consistent interface across all node registries

    Example:
        ```python
        from omnibase_core.container import ModelONEXContainer
        from omnibase_infra.nodes.node_ledger_projection_compute.registry import (
            RegistryInfraLedgerProjection,
        )

        container = ModelONEXContainer()
        node = RegistryInfraLedgerProjection.create_node(container)
        intent = node.compute(event_message)
        ```
    """

    @staticmethod
    def get_node_class() -> type[NodeLedgerProjectionCompute]:
        """Return the node class for DI resolution.

        This method enables DI containers to resolve the node class type
        without importing the node module directly, supporting lazy loading
        and circular import prevention.

        Returns:
            The NodeLedgerProjectionCompute class type.

        Example:
            ```python
            node_class = RegistryInfraLedgerProjection.get_node_class()
            assert node_class is NodeLedgerProjectionCompute
            ```
        """
        return NodeLedgerProjectionCompute

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodeLedgerProjectionCompute:
        """Create a NodeLedgerProjectionCompute instance with the given container.

        Factory method for creating properly-configured node instances.
        Prefer this method over direct instantiation for consistency
        and future extensibility.

        Args:
            container: ONEX dependency injection container.

        Returns:
            Configured NodeLedgerProjectionCompute instance.

        Example:
            ```python
            from omnibase_core.container import ModelONEXContainer

            container = ModelONEXContainer()
            node = RegistryInfraLedgerProjection.create_node(container)
            ```
        """
        return NodeLedgerProjectionCompute(container)


__all__ = [
    "NodeLedgerProjectionCompute",
    "RegistryInfraLedgerProjection",
]

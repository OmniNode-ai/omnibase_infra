# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeContractRegistryReducer dependencies.

This registry provides dependency injection configuration for the
NodeContractRegistryReducer node, following ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.contract_registry_reducer.node import (
        NodeContractRegistryReducer,
    )
    from omnibase_infra.nodes.contract_registry_reducer.reducer import (
        ContractRegistryReducer,
    )


class RegistryInfraContractRegistryReducer:
    """Registry for NodeContractRegistryReducer dependency injection.

    Why a class instead of a function?
        ONEX registry pattern (CLAUDE.md) requires registry classes with
        the naming convention ``RegistryInfra<NodeName>``. This enables:

        - **Future extension**: Additional factory methods can be added
          (e.g., ``create_reducer_with_cache()``, ``create_test_reducer()``)
        - **Service registry resolution**: Classes can be registered in the
          ONEX service registry for container-based resolution
        - **Consistent pattern**: All node registries follow the same class-based
          structure, making the codebase predictable and navigable
        - **Container lifecycle management**: The registry can implement caching,
          scoping, or other lifecycle behaviors in the future

    Provides factory methods for creating NodeContractRegistryReducer instances
    with properly configured dependencies from the ONEX container.

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.contract_registry_reducer.registry import (
            RegistryInfraContractRegistryReducer,
        )

        # Create container and registry
        container = ModelONEXContainer()
        registry = RegistryInfraContractRegistryReducer(container)

        # Create node instance
        node = registry.create_node()

        # Or create pure function reducer directly
        reducer = registry.create_reducer()

        # Use reducer
        result = reducer.reduce(state, event, metadata)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container
        """
        self._container = container

    def create_node(self) -> NodeContractRegistryReducer:
        """Create a NodeContractRegistryReducer instance.

        Returns:
            Configured NodeContractRegistryReducer instance.
        """
        from omnibase_infra.nodes.contract_registry_reducer.node import (
            NodeContractRegistryReducer,
        )

        return NodeContractRegistryReducer(self._container)

    def create_reducer(self) -> ContractRegistryReducer:
        """Create a pure function ContractRegistryReducer instance.

        The pure reducer implements the reduce(state, event, metadata) pattern
        without ONEX container dependencies, suitable for unit testing and
        standalone use.

        Returns:
            ContractRegistryReducer instance (pure function reducer).
        """
        from omnibase_infra.nodes.contract_registry_reducer.reducer import (
            ContractRegistryReducer,
        )

        return ContractRegistryReducer()


__all__ = ["RegistryInfraContractRegistryReducer"]

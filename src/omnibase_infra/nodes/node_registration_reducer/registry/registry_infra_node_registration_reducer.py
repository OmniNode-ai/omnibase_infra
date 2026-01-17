# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeRegistrationReducer dependencies.

This registry provides dependency injection configuration for the
NodeRegistrationReducer node, following ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_registration_reducer.node import (
        NodeRegistrationReducer,
    )


class RegistryInfraNodeRegistrationReducer:
    """Registry for NodeRegistrationReducer dependency injection.

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

    Provides factory methods for creating NodeRegistrationReducer instances
    with properly configured dependencies from the ONEX container.

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_registration_reducer.registry import (
            RegistryInfraNodeRegistrationReducer,
        )

        # Create container and registry
        container = ModelONEXContainer()
        registry = RegistryInfraNodeRegistrationReducer(container)

        # Create reducer instance
        reducer = registry.create_reducer()

        # Use reducer
        result = await reducer.process(input_data)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container
        """
        self._container = container

    def create_reducer(self) -> NodeRegistrationReducer:
        """Create a NodeRegistrationReducer instance.

        Returns:
            Configured NodeRegistrationReducer instance.
        """
        from omnibase_infra.nodes.node_registration_reducer.node import (
            NodeRegistrationReducer,
        )

        return NodeRegistrationReducer(self._container)


__all__ = ["RegistryInfraNodeRegistrationReducer"]

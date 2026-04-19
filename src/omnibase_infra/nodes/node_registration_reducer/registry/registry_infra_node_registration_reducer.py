# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry for NodeRegistrationReducer dependencies.

This registry provides dependency injection configuration for the
NodeRegistrationReducer node, following ONEX container-based DI pattern.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
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

    # ---------------------------------------------------------------------
    # Declaration vs materialization (OMN-9198, HandlerResolver Phase 1)
    # ---------------------------------------------------------------------
    #
    # This reducer node declares NO event handlers -- its contract.yaml has
    # no ``handler_routing`` section; reduction is driven by the pure
    # ``delta(state, event) -> (new_state, intents)`` pattern on the node
    # itself. Therefore the explicit-dependency shape is an empty mapping.
    #
    # The classmethod is still exposed so the HandlerResolver auto-wiring
    # path (Task 3) and Task 10 validators can treat every per-node
    # registry uniformly -- every ``RegistryInfra<Node>`` class answers
    # ``declare_explicit_dependencies()`` and the resolver doesn't need to
    # special-case handler-less nodes.
    # ---------------------------------------------------------------------

    _EXPLICIT_DEPENDENCY_SHAPE: Mapping[str, tuple[str, ...]] = MappingProxyType({})

    @classmethod
    def declare_explicit_dependencies(cls) -> Mapping[str, tuple[str, ...]]:
        """Return the declarative explicit-dependency shape for this node.

        The reducer declares no event handlers (its contract.yaml has no
        ``handler_routing`` section), so the shape is an empty mapping.
        The method exists for uniformity across all per-node registries
        so the HandlerResolver auto-wiring path (Task 3) can treat every
        registry identically at contract-discovery time.

        Returns:
            An immutable empty mapping.

        See Also:
            ``docs/plans/2026-04-18-handler-resolver-architecture.md`` Task 6.
        """
        return cls._EXPLICIT_DEPENDENCY_SHAPE

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

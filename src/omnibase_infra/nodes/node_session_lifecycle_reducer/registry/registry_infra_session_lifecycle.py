# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeSessionLifecycleReducer dependencies.

Provides dependency injection configuration for the session lifecycle
reducer node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_session_lifecycle_reducer.node import (
        NodeSessionLifecycleReducer,
    )


class RegistryInfraSessionLifecycle:
    """Registry for NodeSessionLifecycleReducer dependency injection.

    Provides factory methods for creating NodeSessionLifecycleReducer
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraSessionLifecycle(container)
        >>> reducer = registry.create_reducer()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_reducer(self) -> NodeSessionLifecycleReducer:
        """Create a NodeSessionLifecycleReducer instance.

        Returns:
            Configured NodeSessionLifecycleReducer instance.
        """
        from omnibase_infra.nodes.node_session_lifecycle_reducer.node import (
            NodeSessionLifecycleReducer,
        )

        return NodeSessionLifecycleReducer(self._container)


__all__: list[str] = ["RegistryInfraSessionLifecycle"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodePatternLifecycleEffect dependencies.

Provides dependency injection configuration for the pattern lifecycle
effect node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_pattern_lifecycle_effect.node import (
        NodePatternLifecycleEffect,
    )


class RegistryInfraPatternLifecycle:
    """Registry for NodePatternLifecycleEffect dependency injection.

    Provides factory methods for creating NodePatternLifecycleEffect
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraPatternLifecycle(container)
        >>> effect = registry.create_effect()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_effect(self) -> NodePatternLifecycleEffect:
        """Create a NodePatternLifecycleEffect instance.

        Returns:
            Configured NodePatternLifecycleEffect instance.
        """
        from omnibase_infra.nodes.node_pattern_lifecycle_effect.node import (
            NodePatternLifecycleEffect,
        )

        return NodePatternLifecycleEffect(self._container)


__all__: list[str] = ["RegistryInfraPatternLifecycle"]

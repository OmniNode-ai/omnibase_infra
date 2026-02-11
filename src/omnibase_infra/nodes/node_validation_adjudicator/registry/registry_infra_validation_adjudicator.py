# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeValidationAdjudicator dependencies.

Provides dependency injection configuration for the validation adjudicator
reducer node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_validation_adjudicator.node import (
        NodeValidationAdjudicator,
    )


class RegistryInfraValidationAdjudicator:
    """Registry for NodeValidationAdjudicator dependency injection.

    Provides factory methods for creating NodeValidationAdjudicator
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraValidationAdjudicator(container)
        >>> adjudicator = registry.create_adjudicator()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_adjudicator(self) -> NodeValidationAdjudicator:
        """Create a NodeValidationAdjudicator instance.

        Returns:
            Configured NodeValidationAdjudicator instance.
        """
        from omnibase_infra.nodes.node_validation_adjudicator.node import (
            NodeValidationAdjudicator,
        )

        return NodeValidationAdjudicator(self._container)


__all__: list[str] = ["RegistryInfraValidationAdjudicator"]

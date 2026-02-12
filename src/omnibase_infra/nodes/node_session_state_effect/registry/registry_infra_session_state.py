# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeSessionStateEffect dependencies.

Provides dependency injection configuration for the session state
effect node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_session_state_effect.node import (
        NodeSessionStateEffect,
    )


class RegistryInfraSessionState:
    """Registry for NodeSessionStateEffect dependency injection.

    Provides factory methods for creating NodeSessionStateEffect instances
    with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraSessionState(container)
        >>> node = registry.create_node()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_node(self) -> NodeSessionStateEffect:
        """Create a NodeSessionStateEffect instance.

        Returns:
            Configured NodeSessionStateEffect instance.
        """
        from omnibase_infra.nodes.node_session_state_effect.node import (
            NodeSessionStateEffect,
        )

        return NodeSessionStateEffect(self._container)

    @staticmethod
    def default_state_dir() -> Path:
        """Return the default session state directory.

        Returns:
            Path to ``~/.claude/state``.
        """
        return Path.home() / ".claude" / "state"


__all__: list[str] = ["RegistryInfraSessionState"]

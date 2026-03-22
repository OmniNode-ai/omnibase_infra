# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""DI registration for ProtocolTopicRegistry.

Registers ServiceTopicRegistry with ModelONEXContainer's service registry
as the default ProtocolTopicRegistry implementation.

Usage:
    >>> from omnibase_infra.topics.registry_topic_registry import (
    ...     RegistryTopicRegistry,
    ... )
    >>> await RegistryTopicRegistry.register(container)

    Or via wire_infrastructure_services() which calls this automatically.

Related:
    - OMN-5839: Topic registry consolidation epic
    - ProtocolTopicRegistry: Protocol interface
    - ServiceTopicRegistry: Concrete implementation

.. versionadded:: 0.24.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)


class RegistryTopicRegistry:
    """DI registration for the topic registry.

    Provides a static ``register()`` method following the established
    pattern from ``RegistryInfraServiceDiscovery``.

    .. versionadded:: 0.24.0
    """

    @staticmethod
    async def register(container: ModelONEXContainer) -> None:
        """Register ServiceTopicRegistry in the container.

        Creates a ``ServiceTopicRegistry.from_defaults()`` instance and
        registers it as the ``ProtocolTopicRegistry`` implementation.

        Safe to call multiple times -- subsequent calls replace the
        registration with an equivalent instance.

        Args:
            container: ONEX dependency injection container.

        .. versionadded:: 0.24.0
        """
        if container.service_registry is None:
            return

        from omnibase_core.enums import EnumInjectionScope
        from omnibase_infra.protocols import ProtocolTopicRegistry
        from omnibase_infra.topics.service_topic_registry import (
            ServiceTopicRegistry,
        )

        registry = ServiceTopicRegistry.from_defaults()

        await container.service_registry.register_instance(
            interface=ProtocolTopicRegistry,  # type: ignore[type-abstract]
            instance=registry,
            scope=EnumInjectionScope.GLOBAL,
        )

        logger.debug("Registered ProtocolTopicRegistry in container (global scope)")


__all__ = ["RegistryTopicRegistry"]

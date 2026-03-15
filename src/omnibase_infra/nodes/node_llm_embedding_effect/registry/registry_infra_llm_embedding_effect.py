# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry for LLM Embedding Effect Node Dependencies.

RegistryInfraLlmEmbeddingEffect for registering
embedding handler dependencies with the ONEX container.

Architecture:
    RegistryInfraLlmEmbeddingEffect handles dependency injection setup
    for the NodeLlmEmbeddingEffect node:
    - Registers handler implementations (OpenAI-compatible)
    - Provides factory methods for handler instantiation

Usage:
    The registry is typically called during application bootstrap:

    .. code-block:: python

        from omnibase_infra.nodes.node_llm_embedding_effect.registry import (
            RegistryInfraLlmEmbeddingEffect,
        )

        container = ModelONEXContainer()
        await RegistryInfraLlmEmbeddingEffect.register_openai_compatible(container)

Related:
    - NodeLlmEmbeddingEffect: Node that consumes registered dependencies
    - HandlerEmbeddingOpenaiCompatible: OpenAI-compatible handler
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)


class RegistryInfraLlmEmbeddingEffect:
    """Registry for LLM embedding effect node dependencies.

    Provides static methods for registering embedding handler
    dependencies with the ONEX container.

    Class Methods:
        register: Register with default/environment-based configuration.
        register_openai_compatible: Register OpenAI-compatible handler.
    """

    @staticmethod
    def register(container: ModelONEXContainer) -> None:  # stub-ok
        """Register embedding handler dependencies with default configuration.

        Args:
            container: ONEX dependency injection container.

        Raises:
            NotImplementedError: Always. Use ``register_openai_compatible()``
                instead.
        """
        raise NotImplementedError(
            "register() is not implemented. Use register_openai_compatible() instead."
        )

    @staticmethod
    async def register_openai_compatible(
        container: ModelONEXContainer,
        target_name: str = "openai-embedding",
    ) -> None:
        """Register an OpenAI-compatible embedding handler.

        Creates and registers a ``HandlerEmbeddingOpenaiCompatible``
        instance with the given target name using the container's
        service registry API.

        Args:
            container: ONEX dependency injection container.
            target_name: Identifier for the target (used in error context
                and logging). Default: ``"openai-embedding"``.
        """
        from omnibase_core.enums import EnumInjectionScope
        from omnibase_infra.nodes.node_llm_embedding_effect.handlers import (
            HandlerEmbeddingOpenaiCompatible,
        )

        handler = HandlerEmbeddingOpenaiCompatible(target_name=target_name)

        if container.service_registry is None:
            return

        await container.service_registry.register_instance(
            interface=HandlerEmbeddingOpenaiCompatible,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )
        logger.info(
            "Registered OpenAI-compatible embedding handler: %s",
            target_name,
        )


__all__ = ["RegistryInfraLlmEmbeddingEffect"]

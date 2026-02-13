# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for LLM Embedding Effect Node Dependencies.

This module provides RegistryInfraLlmEmbeddingEffect for registering
embedding handler dependencies with the ONEX container.

Architecture:
    RegistryInfraLlmEmbeddingEffect handles dependency injection setup
    for the NodeLlmEmbeddingEffect node:
    - Registers handler implementations (OpenAI-compatible, Ollama)
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
    - HandlerEmbeddingOllama: Ollama handler
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
        register_ollama: Register Ollama handler.
    """

    @staticmethod
    def register(container: ModelONEXContainer) -> None:
        """Register embedding handler dependencies with default configuration.

        Args:
            container: ONEX dependency injection container.

        Raises:
            NotImplementedError: Always. Use ``register_openai_compatible()``
                or ``register_ollama()`` instead.
        """
        raise NotImplementedError(
            "register() is not implemented. "
            "Use register_openai_compatible() or register_ollama() instead."
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

    @staticmethod
    async def register_ollama(
        container: ModelONEXContainer,
        target_name: str = "ollama-embedding",
    ) -> None:
        """Register an Ollama embedding handler.

        Creates and registers a ``HandlerEmbeddingOllama`` instance
        with the given target name using the container's service
        registry API.

        Args:
            container: ONEX dependency injection container.
            target_name: Identifier for the target (used in error context
                and logging). Default: ``"ollama-embedding"``.
        """
        from omnibase_core.enums import EnumInjectionScope
        from omnibase_infra.nodes.node_llm_embedding_effect.handlers import (
            HandlerEmbeddingOllama,
        )

        handler = HandlerEmbeddingOllama(target_name=target_name)

        if container.service_registry is None:
            return

        await container.service_registry.register_instance(
            interface=HandlerEmbeddingOllama,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )
        logger.info(
            "Registered Ollama embedding handler: %s",
            target_name,
        )


__all__ = ["RegistryInfraLlmEmbeddingEffect"]

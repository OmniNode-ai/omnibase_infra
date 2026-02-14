# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for LLM Inference Effect Node Dependencies.

This module provides RegistryInfraLlmInferenceEffect for registering
inference handler dependencies with the ONEX container and creating
configured node instances.

Architecture:
    RegistryInfraLlmInferenceEffect handles dependency injection setup
    for the NodeLlmInferenceEffect node:
    - Validates required protocols before node construction
    - Registers handler implementations (OpenAI-compatible, Ollama)
    - Provides factory methods for handler instantiation

Usage:
    The registry is typically called during application bootstrap:

    .. code-block:: python

        from omnibase_infra.nodes.node_llm_inference_effect.registry import (
            RegistryInfraLlmInferenceEffect,
        )

        container = ModelONEXContainer()
        await RegistryInfraLlmInferenceEffect.register_ollama(container)

Related:
    - NodeLlmInferenceEffect: Node that consumes registered dependencies
    - HandlerLlmOpenaiCompatible: OpenAI-compatible inference handler
    - HandlerLlmOllama: Ollama inference handler
    - OMN-2111: Phase 11 node assembly

.. versionadded:: 0.8.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.mixins import MixinLlmHttpTransport
    from omnibase_infra.nodes.node_llm_inference_effect.node import (
        NodeLlmInferenceEffect,
    )

logger = logging.getLogger(__name__)


def _create_transport_adapter(
    target_name: str = "openai-inference",
) -> MixinLlmHttpTransport:
    """Create a transport adapter for HandlerLlmOpenaiCompatible.

    HandlerLlmOpenaiCompatible requires a transport instance (any object
    that provides ``_execute_llm_http_call``) via constructor injection.
    This factory creates a proper MixinLlmHttpTransport subclass instance
    to serve as that transport.

    This function is private to the registry module and should not be
    used directly outside of ``RegistryInfraLlmInferenceEffect``.

    Args:
        target_name: Identifier for the target used in error context
            and logging.

    Returns:
        A MixinLlmHttpTransport instance providing ``_execute_llm_http_call``.
    """
    from omnibase_infra.mixins import MixinLlmHttpTransport

    class _TransportInstance(MixinLlmHttpTransport):
        def __init__(self, name: str) -> None:
            self._init_llm_http_transport(target_name=name)

    return _TransportInstance(target_name)


class RegistryInfraLlmInferenceEffect:
    """Infrastructure registry for NodeLlmInferenceEffect.

    Provides dependency resolution, factory methods, and handler
    registration for creating properly configured NodeLlmInferenceEffect
    instances.

    This registry follows the ONEX infrastructure registry pattern:
        - Factory method for node creation with container injection
        - Protocol validation before node construction
        - Handler registration methods for each supported backend
        - Node type classification for routing decisions
        - Capability listing for service discovery

    Class Methods:
        create: Create node with validated container dependencies.
        register_openai_compatible: Register OpenAI-compatible handler.
        register_ollama: Register Ollama handler.

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> from omnibase_infra.nodes.node_llm_inference_effect.registry import (
        ...     RegistryInfraLlmInferenceEffect,
        ... )
        >>> container = ModelONEXContainer()
        >>> effect = RegistryInfraLlmInferenceEffect.create(container)

    .. versionadded:: 0.8.0
    """

    @staticmethod
    def create(container: ModelONEXContainer) -> NodeLlmInferenceEffect:
        """Create a NodeLlmInferenceEffect instance with resolved dependencies.

        Validates that all required protocols are resolvable from the
        container before constructing the node. This ensures errors
        surface at construction time rather than at runtime.

        Args:
            container: ONEX dependency injection container.

        Returns:
            Configured NodeLlmInferenceEffect instance ready for operation.

        Raises:
            OnexError: If required protocols are not registered in container.
                Specifically raised when ``MixinLlmHttpTransport`` cannot be
                resolved from the container's service registry.

        .. versionadded:: 0.8.0
        """
        from omnibase_infra.mixins import MixinLlmHttpTransport
        from omnibase_infra.nodes.node_llm_inference_effect.node import (
            NodeLlmInferenceEffect,
        )

        # Validate required protocols are resolvable
        if container.service_registry is not None:
            try:
                container.get_service(MixinLlmHttpTransport)
            except Exception as exc:
                from omnibase_core.errors import OnexError

                msg = (
                    f"Required protocol '{MixinLlmHttpTransport.__name__}' "
                    f"is not registered in the container. "
                    f"Call register_openai_compatible() or register_ollama() "
                    f"before creating the node."
                )
                raise OnexError(msg) from exc

        return NodeLlmInferenceEffect(container)

    @staticmethod
    async def register_openai_compatible(
        container: ModelONEXContainer,
        target_name: str = "openai-inference",
    ) -> None:
        """Register an OpenAI-compatible inference handler.

        Creates and registers a ``HandlerLlmOpenaiCompatible`` instance
        with a transport adapter, using the container's service registry
        API.

        Args:
            container: ONEX dependency injection container.
            target_name: Identifier for the target (used in error context
                and logging). Default: ``"openai-inference"``.
        """
        from omnibase_core.enums import EnumInjectionScope
        from omnibase_infra.mixins import MixinLlmHttpTransport
        from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
            HandlerLlmOpenaiCompatible,
        )

        transport = _create_transport_adapter(target_name=target_name)
        handler = HandlerLlmOpenaiCompatible(transport=transport)

        if container.service_registry is None:
            return

        await container.service_registry.register_instance(
            interface=HandlerLlmOpenaiCompatible,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )
        await container.service_registry.register_instance(
            interface=MixinLlmHttpTransport,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )
        logger.info(
            "Registered OpenAI-compatible inference handler: %s",
            target_name,
        )

    @staticmethod
    async def register_ollama(
        container: ModelONEXContainer,
        target_name: str = "ollama-inference",
    ) -> None:
        """Register an Ollama inference handler.

        Creates and registers a ``HandlerLlmOllama`` instance
        with the given target name using the container's service
        registry API.

        Args:
            container: ONEX dependency injection container.
            target_name: Identifier for the target (used in error context
                and logging). Default: ``"ollama-inference"``.
        """
        from omnibase_core.enums import EnumInjectionScope
        from omnibase_infra.mixins import MixinLlmHttpTransport
        from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
            HandlerLlmOllama,
        )

        handler = HandlerLlmOllama(target_name=target_name)

        if container.service_registry is None:
            return

        await container.service_registry.register_instance(
            interface=HandlerLlmOllama,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )
        await container.service_registry.register_instance(
            interface=MixinLlmHttpTransport,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )
        logger.info(
            "Registered Ollama inference handler: %s",
            target_name,
        )

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Get list of protocols required by this node.

        Returns:
            List of protocol class names required for node operation.

        .. versionadded:: 0.8.0
        """
        return [
            "MixinLlmHttpTransport",
        ]

    @staticmethod
    def get_node_type() -> str:
        """Get the node type classification.

        Returns:
            Node type string matching contract.yaml ("EFFECT_GENERIC").

        .. versionadded:: 0.8.0
        """
        return "EFFECT_GENERIC"

    @staticmethod
    def get_node_name() -> str:
        """Get the canonical node name.

        Returns:
            The node name as defined in contract.yaml.

        .. versionadded:: 0.8.0
        """
        return "node_llm_inference_effect"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Get list of capabilities provided by this node.

        Returns:
            List of capability identifiers.

        .. versionadded:: 0.8.0
        """
        return [
            "openai_compatible_inference",
            "ollama_inference",
            "chat_completion",
            "tool_calling",
            "circuit_breaker_protection",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Get list of operations supported by this node.

        Returns:
            List of operation identifiers as defined in contract.yaml.

        .. versionadded:: 0.8.0
        """
        return [
            "inference.openai_compatible",
            "inference.ollama",
        ]

    @staticmethod
    def get_backends() -> list[str]:
        """Get list of backend types this node interacts with.

        Returns:
            List of backend/provider identifiers.

        .. versionadded:: 0.8.0
        """
        return ["openai_compatible", "ollama"]


__all__ = ["RegistryInfraLlmInferenceEffect"]

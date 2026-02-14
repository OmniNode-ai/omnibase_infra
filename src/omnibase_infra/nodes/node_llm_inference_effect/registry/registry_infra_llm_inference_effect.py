# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeLlmInferenceEffect infrastructure dependencies.

This registry provides factory methods for creating NodeLlmInferenceEffect
instances with their required dependencies resolved from the container.

Following ONEX naming conventions:
    - File: registry_infra_<node_name>.py
    - Class: RegistryInfra<NodeName>

Related:
    - contract.yaml: Node contract defining operations and dependencies
    - node.py: Declarative node implementation
    - handlers/: Provider-specific handler implementations
    - OMN-2111: Phase 11 node assembly

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_llm_inference_effect.node import (
        NodeLlmInferenceEffect,
    )


class RegistryInfraLlmInferenceEffect:
    """Infrastructure registry for NodeLlmInferenceEffect.

    Provides dependency resolution and factory methods for creating
    properly configured NodeLlmInferenceEffect instances.

    This registry follows the ONEX infrastructure registry pattern:
        - Factory method for node creation with container injection
        - Protocol requirements documentation for container validation
        - Node type classification for routing decisions
        - Capability listing for service discovery

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

        Args:
            container: ONEX dependency injection container.

        Returns:
            Configured NodeLlmInferenceEffect instance ready for operation.

        Raises:
            OnexError: If required protocols are not registered in container.

        .. versionadded:: 0.8.0
        """
        from omnibase_infra.nodes.node_llm_inference_effect.node import (
            NodeLlmInferenceEffect,
        )

        return NodeLlmInferenceEffect(container)

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
            Node type string ("EFFECT").

        .. versionadded:: 0.8.0
        """
        return "EFFECT"

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

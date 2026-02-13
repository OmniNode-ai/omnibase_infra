# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM Embedding Effect Node - Declarative shell.

This node provides embedding generation operations via pluggable handlers
(OpenAI-compatible, Ollama). All behavior is defined in contract.yaml.

Supported Operations:
    - embedding.openai_compatible: Generate embeddings via OpenAI-compatible API
    - embedding.ollama: Generate embeddings via Ollama API

Declarative Pattern:
    This node follows ONEX declarative conventions:
    - Extends NodeEffect from omnibase_core
    - All behavior defined in contract.yaml
    - No custom routing logic in Python code
    - Handler wired externally via container injection

Related:
    - contract.yaml: ONEX contract with operations and I/O definitions
    - HandlerEmbeddingOpenaiCompatible: OpenAI-compatible handler
    - HandlerEmbeddingOllama: Ollama handler
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeLlmEmbeddingEffect(NodeEffect):
    """Effect node for LLM embedding generation.

    Provides embedding generation via pluggable handlers supporting
    OpenAI-compatible and Ollama APIs.

    This is a declarative node - all behavior is driven by contract.yaml.
    The handler implementation is resolved from the container at runtime.

    Attributes:
        container: ONEX dependency injection container providing access
            to the configured embedding handler implementation.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the LLM embedding effect node.

        Args:
            container: ONEX dependency injection container providing
                access to infrastructure dependencies including the
                configured embedding handler implementation.
        """
        super().__init__(container)


__all__ = ["NodeLlmEmbeddingEffect"]

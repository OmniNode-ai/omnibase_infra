# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM Embedding Effect Node - Vector embedding generation.

This module provides NodeLlmEmbeddingEffect, an effect node for generating
vector embeddings via pluggable handler backends.

Supported Operations:
    - ``embedding.openai_compatible``: OpenAI-compatible ``/v1/embeddings`` endpoint
    - ``embedding.ollama``: Ollama ``/api/embed`` endpoint

Node:
    NodeLlmEmbeddingEffect: Declarative effect node for embedding generation.

Models:
    ModelLlmEmbeddingRequest: Input model with texts validation (1-2048 items).
    ModelLlmEmbeddingResponse: Output model with dimension uniformity validator.

Handlers:
    HandlerEmbeddingOpenaiCompatible: Handler for OpenAI-compatible endpoints.
    HandlerEmbeddingOllama: Handler for Ollama endpoints.

Registry:
    RegistryInfraLlmEmbeddingEffect: DI registry for node dependencies.

Constants:
    ALLOWED_EMBEDDING_OPERATIONS: Frozenset of valid embedding operation strings.

Example:
    .. code-block:: python

        from omnibase_infra.nodes.node_llm_embedding_effect import (
            NodeLlmEmbeddingEffect,
            ModelLlmEmbeddingRequest,
            HandlerEmbeddingOpenaiCompatible,
        )

        handler = HandlerEmbeddingOpenaiCompatible(target_name="vllm-gte")
        request = ModelLlmEmbeddingRequest(
            base_url="http://192.168.86.201:8002",
            model="gte-qwen2-1.5b",
            texts=("Hello, world!",),
        )
        response = await handler.execute(request)

Related:
    - OMN-2112: Phase 12 embedding node

.. versionadded:: 0.7.0
"""

import typing

from .handlers import (
    HandlerEmbeddingOllama,
    HandlerEmbeddingOpenaiCompatible,
)
from .models import (
    ModelLlmEmbeddingRequest,
    ModelLlmEmbeddingResponse,
)
from .node import (
    NodeLlmEmbeddingEffect,
)
from .registry import (
    RegistryInfraLlmEmbeddingEffect,
)

ALLOWED_EMBEDDING_OPERATIONS: typing.Final[frozenset[str]] = frozenset(
    {
        "embedding.openai_compatible",
        "embedding.ollama",
    }
)

__all__ = [
    # Node
    "NodeLlmEmbeddingEffect",
    # Models
    "ModelLlmEmbeddingRequest",
    "ModelLlmEmbeddingResponse",
    # Handlers
    "HandlerEmbeddingOpenaiCompatible",
    "HandlerEmbeddingOllama",
    # Registry
    "RegistryInfraLlmEmbeddingEffect",
    # Constants
    "ALLOWED_EMBEDDING_OPERATIONS",
]

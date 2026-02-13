# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for the LLM Embedding Effect Node.

This module exports embedding handlers for different LLM providers:

Classes:
    HandlerEmbeddingOpenaiCompatible: OpenAI-compatible embedding handler.
    HandlerEmbeddingOllama: Ollama embedding handler.

Ticket: OMN-2112
"""

from omnibase_infra.nodes.node_llm_embedding_effect.handlers.handler_embedding_ollama import (
    HandlerEmbeddingOllama,
)
from omnibase_infra.nodes.node_llm_embedding_effect.handlers.handler_embedding_openai_compatible import (
    HandlerEmbeddingOpenaiCompatible,
)

__all__: list[str] = [
    "HandlerEmbeddingOllama",
    "HandlerEmbeddingOpenaiCompatible",
]

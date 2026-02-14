# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler implementations for the LLM inference effect node."""

from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_ollama import (
    HandlerLlmOllama,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)

__all__ = ["HandlerLlmOllama", "HandlerLlmOpenaiCompatible"]

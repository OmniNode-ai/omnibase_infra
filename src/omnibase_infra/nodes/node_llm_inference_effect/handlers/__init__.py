# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for the LLM Inference Effect node.

Exports:
    HandlerLlmOpenaiCompatible: OpenAI wire-format inference handler.
"""

from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)

__all__: list[str] = ["HandlerLlmOpenaiCompatible"]

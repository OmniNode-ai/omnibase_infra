# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""LLM Inference Effect Node package.

Provides the declarative effect node for LLM inference operations
and its associated handlers for provider-specific communication.

Exports:
    NodeLlmInferenceEffect: Declarative effect node (zero custom logic)
    ALLOWED_INFERENCE_OPERATIONS: Valid operation identifiers for routing
    HandlerLlmOllama: Ollama-native inference handler
    HandlerLlmOpenaiCompatible: OpenAI wire-format inference handler
    RegistryInfraLlmInferenceEffect: Factory and metadata registry

Related:
    - OMN-2107: OpenAI-compatible handler
    - OMN-2108: Ollama handler
    - OMN-2111: Node assembly
"""

from __future__ import annotations

from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
    HandlerLlmOllama,
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.node import (
    NodeLlmInferenceEffect,
)
from omnibase_infra.nodes.node_llm_inference_effect.registry import (
    RegistryInfraLlmInferenceEffect,
)

# Derived from registry (which reads contract.yaml) so there is a single
# source of truth for supported operations.
ALLOWED_INFERENCE_OPERATIONS: frozenset[str] = frozenset(
    RegistryInfraLlmInferenceEffect.get_supported_operations()
)

__all__: list[str] = [
    "ALLOWED_INFERENCE_OPERATIONS",
    "HandlerLlmOllama",
    "HandlerLlmOpenaiCompatible",
    "NodeLlmInferenceEffect",
    "RegistryInfraLlmInferenceEffect",
]

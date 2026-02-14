# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Models for the LLM Inference Effect node.

Exports node-specific models and re-exports shared effect models
for convenience.

Node-specific:
    ModelLlmInferenceRequest: Node-local input request model

Re-exported from shared effect models:
    ModelLlmInferenceResponse: Canonical inference response
    ModelLlmMessage: Chat message model
    ModelLlmUsage: Token usage tracking
    ModelLlmToolCall: Tool call from model response
    ModelLlmToolChoice: Tool selection constraint
    ModelLlmToolDefinition: Tool definition for request
    ModelLlmFunctionCall: Function invocation from LLM
    ModelLlmFunctionDef: Function schema definition
    ModelBackendResult: Backend operation outcome
"""

from __future__ import annotations

from omnibase_infra.models import ModelBackendResult
from omnibase_infra.nodes.effects.models import (
    ModelLlmFunctionCall,
    ModelLlmFunctionDef,
    ModelLlmInferenceResponse,
    ModelLlmMessage,
    ModelLlmToolCall,
    ModelLlmToolChoice,
    ModelLlmToolDefinition,
    ModelLlmUsage,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

__all__: list[str] = [
    "ModelBackendResult",
    "ModelLlmFunctionCall",
    "ModelLlmFunctionDef",
    "ModelLlmInferenceRequest",
    "ModelLlmInferenceResponse",
    "ModelLlmMessage",
    "ModelLlmToolCall",
    "ModelLlmToolChoice",
    "ModelLlmToolDefinition",
    "ModelLlmUsage",
]

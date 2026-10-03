# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Models for the LLM Inference Effect node.

Exports node-specific models and re-exports shared effect models
for convenience.

Node-specific:
    ModelLlmInferenceRequest: The single input request model, imported by the
        handlers and named by ``input_model`` in contract.yaml (OMN-17104).

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
from omnibase_infra.models.llm import (
    ModelLlmFunctionCall,
    ModelLlmFunctionDef,
    ModelLlmInferenceResponse,
    ModelLlmMessage,
    ModelLlmToolCall,
    ModelLlmToolChoice,
    ModelLlmToolDefinition,
    ModelLlmUsage,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_call_completed_event import (
    ModelLlmCallCompletedEvent,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_call_completed_infra_event import (
    ModelLlmCallCompletedInfraEvent,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_command import (
    ModelLlmInferenceCommand,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

__all__: list[str] = [
    "ModelBackendResult",
    "ModelLlmCallCompletedEvent",
    "ModelLlmCallCompletedInfraEvent",
    "ModelLlmFunctionCall",
    "ModelLlmFunctionDef",
    "ModelLlmInferenceCommand",
    "ModelLlmInferenceRequest",
    "ModelLlmInferenceResponse",
    "ModelLlmMessage",
    "ModelLlmToolCall",
    "ModelLlmToolChoice",
    "ModelLlmToolDefinition",
    "ModelLlmUsage",
]

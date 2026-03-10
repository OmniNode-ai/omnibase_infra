# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Shared LLM models for inference, embeddings, and tool-calling interactions.

These models are domain-level models used across multiple LLM-related nodes
(node_llm_inference_effect, node_llm_embedding_effect) and are not node-specific.

Available Models:
    - ModelLlmFunctionCall: Concrete function invocation from an LLM
    - ModelLlmFunctionDef: JSON-Schema description of a callable function
    - ModelLlmInferenceRequest: Input model for the LLM inference effect node
    - ModelLlmInferenceResponse: LLM inference output with text XOR tool_calls invariant
    - ModelLlmMessage: Chat message for multi-turn LLM conversations
    - ModelLlmToolCall: Tool call returned by the model
    - ModelLlmToolChoice: Caller constraint on tool selection behaviour
    - ModelLlmToolDefinition: Tool definition sent in request payload
    - ModelLlmUsage: Token-usage summary from an LLM provider

Adapters:
    - to_call_metrics: ModelLlmUsage -> ContractLlmCallMetrics
    - to_usage_normalized: ModelLlmUsage -> ContractLlmUsageNormalized
    - to_usage_raw: ModelLlmUsage -> ContractLlmUsageRaw

Migrated from: omnibase_infra.nodes.effects.models (OMN-3989)
"""

from __future__ import annotations

from omnibase_infra.models.llm.adapter_llm_usage_to_contract import (
    to_call_metrics,
    to_usage_normalized,
    to_usage_raw,
)
from omnibase_infra.models.llm.model_llm_function_call import ModelLlmFunctionCall
from omnibase_infra.models.llm.model_llm_function_def import ModelLlmFunctionDef
from omnibase_infra.models.llm.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
from omnibase_infra.models.llm.model_llm_tool_call import ModelLlmToolCall
from omnibase_infra.models.llm.model_llm_tool_choice import ModelLlmToolChoice
from omnibase_infra.models.llm.model_llm_tool_definition import ModelLlmToolDefinition
from omnibase_infra.models.llm.model_llm_usage import ModelLlmUsage

__all__ = [
    "ModelLlmFunctionCall",
    "ModelLlmFunctionDef",
    "ModelLlmInferenceRequest",
    "ModelLlmInferenceResponse",
    "ModelLlmMessage",
    "ModelLlmToolCall",
    "ModelLlmToolChoice",
    "ModelLlmToolDefinition",
    "ModelLlmUsage",
    "to_call_metrics",
    "to_usage_normalized",
    "to_usage_raw",
]

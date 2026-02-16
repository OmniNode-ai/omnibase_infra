# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Effect nodes.

This module exports models used by Effect layer nodes for external I/O operations.

Available Models:
    - ModelBackendResult: Individual backend operation result (re-exported)
    - ModelEffectIdempotencyConfig: Configuration for effect idempotency store
    - ModelLlmFunctionCall: Concrete function invocation from an LLM
    - ModelLlmFunctionDef: JSON-Schema description of a callable function
    - ModelLlmInferenceRequest: Input model for the LLM inference effect node
    - ModelLlmInferenceResponse: LLM inference output with text XOR tool_calls invariant
    - ModelLlmMessage: Chat message for multi-turn LLM conversations
    - ModelLlmToolCall: Tool call returned by the model
    - ModelLlmToolChoice: Caller constraint on tool selection behaviour
    - ModelLlmToolDefinition: Tool definition sent in request payload
    - ModelLlmUsage: Token-usage summary from an LLM provider
    - ModelRegistryRequest: Registry effect input request
    - ModelRegistryResponse: Dual-backend registry operation response

Converters:
    - to_call_metrics: ModelLlmUsage -> ContractLlmCallMetrics
    - to_usage_normalized: ModelLlmUsage -> ContractLlmUsageNormalized
    - to_usage_raw: ModelLlmUsage -> ContractLlmUsageRaw

Note:
    ModelBackendResult canonical location is omnibase_infra.models.model_backend_result.
    Re-exported here for ergonomic access.
"""

from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.converter_llm_usage_to_contract import (
    to_call_metrics,
    to_usage_normalized,
    to_usage_raw,
)
from omnibase_infra.nodes.effects.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.effects.models.model_llm_function_call import (
    ModelLlmFunctionCall,
)
from omnibase_infra.nodes.effects.models.model_llm_function_def import (
    ModelLlmFunctionDef,
)
from omnibase_infra.nodes.effects.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.nodes.effects.models.model_llm_message import ModelLlmMessage
from omnibase_infra.nodes.effects.models.model_llm_tool_call import ModelLlmToolCall
from omnibase_infra.nodes.effects.models.model_llm_tool_choice import (
    ModelLlmToolChoice,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_definition import (
    ModelLlmToolDefinition,
)
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_infra.nodes.effects.models.model_registry_request import (
    ModelRegistryRequest,
)
from omnibase_infra.nodes.effects.models.model_registry_response import (
    ModelRegistryResponse,
)

__all__ = [
    "ModelBackendResult",
    "ModelEffectIdempotencyConfig",
    "ModelLlmFunctionCall",
    "ModelLlmFunctionDef",
    "ModelLlmInferenceRequest",
    "ModelLlmInferenceResponse",
    "ModelLlmMessage",
    "ModelLlmToolCall",
    "ModelLlmToolChoice",
    "ModelLlmToolDefinition",
    "ModelLlmUsage",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
    "to_call_metrics",
    "to_usage_normalized",
    "to_usage_raw",
]

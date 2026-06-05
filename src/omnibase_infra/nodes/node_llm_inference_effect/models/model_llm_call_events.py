# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""LLM call completion events emitted by the inference effect node."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType

TOPIC_LLM_CALL_COMPLETED = "onex.evt.omniintelligence.llm-call-completed.v1"
TOPIC_LLM_CALL_COMPLETED_INFRA = "onex.evt.omnibase-infra.llm-call-completed.v1"


class ModelLlmCallCompletedEvent(BaseModel):
    """Full LLM metrics event for the omniintelligence cost pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = TOPIC_LLM_CALL_COMPLETED
    schema_version: str = "1.0"
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    latency_ms: float
    usage_raw: dict[str, JsonType] | None = None
    usage_normalized: dict[str, JsonType] | None = None
    usage_is_estimated: bool = False
    input_hash: str = ""
    code_version: str = ""
    contract_version: str = ""
    timestamp_iso: str
    reporting_source: str = ""
    extensions: dict[str, JsonType] = Field(default_factory=dict)
    gpu_seconds: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    compute_usage_source: str | None = None


class ModelLlmCallCompletedInfraEvent(BaseModel):
    """Infra-local LLM completion event for infra projections."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = TOPIC_LLM_CALL_COMPLETED_INFRA
    model_id: str
    endpoint_url: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float
    success: bool = True
    timestamp: str
    gpu_seconds: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    compute_usage_source: str | None = None


__all__: list[str] = [
    "ModelLlmCallCompletedEvent",
    "ModelLlmCallCompletedInfraEvent",
    "TOPIC_LLM_CALL_COMPLETED",
    "TOPIC_LLM_CALL_COMPLETED_INFRA",
]

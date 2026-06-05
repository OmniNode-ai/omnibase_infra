# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infra-local LLM call completion event emitted by the inference effect node."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.topics import SUFFIX_LLM_CALL_COMPLETED_INFRA


class ModelLlmCallCompletedInfraEvent(BaseModel):
    """Infra-local LLM completion event for infra projections."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = SUFFIX_LLM_CALL_COMPLETED_INFRA
    # ONEX_EXCLUDE: pattern_validator - model_id is an LLM model name (e.g. "gemini-2.5-pro"), not a UUID entity reference
    model_id: str = Field(min_length=1)
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


__all__: list[str] = ["ModelLlmCallCompletedInfraEvent"]

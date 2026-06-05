# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Full LLM call completion event emitted by the inference effect node."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.topics import SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED


class ModelLlmCallCompletedEvent(BaseModel):
    """Full LLM metrics event for the omniintelligence cost pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED
    schema_version: str = "1.0"
    # ONEX_EXCLUDE: pattern_validator - model_id is an LLM model name (e.g. "gemini-2.5-pro"), not a UUID entity reference
    model_id: str = Field(min_length=1)
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


__all__: list[str] = ["ModelLlmCallCompletedEvent"]

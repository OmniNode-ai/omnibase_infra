# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Cost metrics captured for a single A/B run variant.

Tracks token usage, wall-clock time, and retry counts to enable
cost-based ROI comparison between baseline and candidate runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCostMetrics(BaseModel):
    """Cost metrics for a single run (baseline or candidate).

    All counters default to zero so that partially-populated metrics
    are valid.

    Note:
        ``total_tokens``, ``prompt_tokens``, and ``completion_tokens`` are
        intentionally independent fields with no consistency validation.
        Some LLM providers return only ``total_tokens`` without a
        prompt/completion breakdown, so enforcing
        ``total == prompt + completion`` would reject valid data.

    Attributes:
        total_tokens: Total tokens consumed. Not enforced as prompt +
            completion sum; some providers report only the total.
        prompt_tokens: Prompt/input tokens consumed.
        completion_tokens: Completion/output tokens consumed.
        wall_time_ms: Wall-clock execution time in milliseconds.
        retry_count: Number of retries needed to complete the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed. Not enforced as prompt + completion sum; some providers report only the total.",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Prompt/input tokens consumed.",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Completion/output tokens consumed.",
    )
    wall_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock execution time in milliseconds.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries needed to complete the run.",
    )


__all__: list[str] = ["ModelCostMetrics"]

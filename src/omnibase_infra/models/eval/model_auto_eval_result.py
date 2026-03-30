# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-eval result model.

Captures the outcome of executing a single eval task, including quality
scores, latency, token usage, and cost.

Related:
    - OMN-6795: Define eval task models and enums
    - ModelAutoEvalTask: The task that produced this result
    - ServiceAutoEvalRunner: The service that executes tasks
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_auto_eval_task_type import EnumAutoEvalTaskType


class ModelAutoEvalResult(BaseModel):
    """Result of executing an autonomous LLM evaluation task.

    Attributes:
        task_id: ID of the task that was executed.
        task_type: Category of evaluation that was performed.
        score: Quality score between 0.0 and 1.0.
        raw_output: The raw LLM response text.
        latency_ms: End-to-end latency in milliseconds.
        tokens_used: Total tokens consumed (prompt + completion).
        cost_usd: Estimated cost in USD for this execution.
        error_message: Non-empty if the task failed.
        completed_at: Timestamp when the evaluation completed.
        model_id: Model identifier that was evaluated.
        endpoint_url: Endpoint URL that was evaluated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    task_id: UUID = Field(
        ...,
        description="ID of the task that was executed.",
    )
    task_type: EnumAutoEvalTaskType = Field(
        ...,
        description="Category of evaluation that was performed.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality score between 0.0 and 1.0.",
    )
    raw_output: str = Field(
        default="",
        description="The raw LLM response text.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end latency in milliseconds.",
    )
    tokens_used: int = Field(
        ...,
        ge=0,
        description="Total tokens consumed (prompt + completion).",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in USD for this execution.",
    )
    error_message: str = Field(
        default="",
        description="Non-empty if the task failed.",
    )
    completed_at: datetime = Field(
        ...,
        description="Timestamp when the evaluation completed.",
    )
    # ONEX_EXCLUDE: pattern_validator - model_id is an LLM model name (e.g. "qwen3-coder-30b"), not a UUID entity reference
    model_id: str = Field(
        ...,
        min_length=1,
        description="Model identifier that was evaluated.",
    )
    endpoint_url: str = Field(
        ...,
        min_length=1,
        description="Endpoint URL that was evaluated.",
    )


__all__: list[str] = ["ModelAutoEvalResult"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-eval task definition model.

Defines a single evaluation task to be executed by ServiceAutoEvalRunner
against a configured LLM endpoint. Each task specifies the evaluation type,
target, prompt, and expected output for scoring.

Related:
    - OMN-6795: Define eval task models and enums
    - EnumAutoEvalTaskType: Task type classification
    - ModelAutoEvalResult: Result produced after execution
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_auto_eval_task_type import EnumAutoEvalTaskType


class ModelAutoEvalTask(BaseModel):
    """Definition of a single autonomous LLM evaluation task.

    Attributes:
        task_id: Unique identifier for this task instance.
        task_type: Category of evaluation to perform.
        prompt: The prompt to send to the LLM endpoint.
        expected_output: Optional reference answer for scoring.
        endpoint_url: LLM endpoint URL to evaluate against.
        model_id: Model identifier at the endpoint.
        max_tokens: Maximum tokens for the LLM response.
        metadata: Arbitrary key-value pairs for observability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    task_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this task instance.",
    )
    task_type: EnumAutoEvalTaskType = Field(
        ...,
        description="Category of evaluation to perform.",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="The prompt to send to the LLM endpoint.",
    )
    expected_output: str | None = Field(
        default=None,
        description="Optional reference answer for scoring.",
    )
    endpoint_url: str = Field(
        ...,
        min_length=1,
        description="LLM endpoint URL to evaluate against.",
    )
    # ONEX_EXCLUDE: pattern_validator - model_id is an LLM model name (e.g. "qwen3-coder-30b"), not a UUID entity reference
    model_id: str = Field(
        ...,
        min_length=1,
        description="Model identifier at the endpoint.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=128_000,
        description="Maximum tokens for the LLM response.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for observability.",
    )


__all__: list[str] = ["ModelAutoEvalTask"]

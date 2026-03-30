# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Eval task model for autonomous off-peak evaluations.

Defines the task request model used by ``ServiceEvalRunner``
to dispatch evaluation tasks to cheap LLM providers.

Related:
    - OMN-6795: Define eval task models and enums
    - OMN-6796: Build eval runner service

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_eval_task_status import EnumEvalTaskStatus
from omnibase_infra.enums.enum_eval_task_type import EnumEvalTaskType


class ModelEvalTask(BaseModel):
    """An evaluation task to be dispatched to a cheap LLM provider.

    Attributes:
        task_id: Unique identifier for this task.
        task_type: The type of evaluation to perform.
        target_repo: Repository name to evaluate (e.g., "omnibase_infra").
        target_path: Specific file or directory path to evaluate.
        llm_model_label: Human-readable LLM model label for evaluation.
        max_tokens: Maximum token budget for this individual task.
        estimated_cost_usd: Estimated cost in USD for this task.
        status: Current task status.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: UUID = Field(default_factory=uuid4, description="Unique task identifier.")
    task_type: EnumEvalTaskType = Field(
        ..., description="Type of evaluation to perform."
    )
    target_repo: str = Field(..., min_length=1, description="Repository to evaluate.")
    target_path: str = Field(
        default="", description="File or directory path within the repo."
    )
    llm_model_label: str = Field(
        default="", description="LLM model to use (empty = auto-select via bifrost)."
    )
    max_tokens: int = Field(
        default=4096, ge=256, le=131072, description="Max tokens for this task."
    )
    estimated_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Estimated cost in USD."
    )
    status: EnumEvalTaskStatus = Field(
        default=EnumEvalTaskStatus.PENDING, description="Task status."
    )


__all__: list[str] = ["ModelEvalTask"]

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Query model for retrieving relevant agent learnings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.agent_learning.enum_learning_match_type import (
    EnumLearningMatchType,
)
from omnibase_infra.models.agent_learning.enum_learning_task_type import (
    EnumLearningTaskType,
)


class ModelAgentLearningQuery(BaseModel):
    """Query for retrieving relevant agent learnings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_type: EnumLearningMatchType = Field(
        default=EnumLearningMatchType.AUTO,
        description="Match strategy: error_signature (high precision), task_context (broad), or auto (try both)",
    )
    error_text: str | None = Field(
        default=None,
        max_length=4000,
        description="Error message to match against (for error_signature and auto match types)",
    )
    repo: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Repository to scope the search to",
    )
    file_paths: tuple[str, ...] = Field(
        default=(),
        description="File paths for context matching",
    )
    task_type: EnumLearningTaskType | None = Field(
        default=None,
        description="Optional task type filter",
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for results",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Query result model for agent learning retrieval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.agent_learning.model_agent_learning_match import (
    ModelAgentLearningMatch,
)


class ModelAgentLearningQueryResult(BaseModel):
    """Result of querying the agent learning store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    matches: tuple[ModelAgentLearningMatch, ...] = Field(
        default=(),
        description="Matched learning records, sorted by combined_score descending",
    )
    query_ms: int = Field(
        ...,
        ge=0,
        description="Total query time in milliseconds",
    )
    error_matches_count: int = Field(
        default=0,
        ge=0,
        description="Number of matches from error signature collection",
    )
    context_matches_count: int = Field(
        default=0,
        ge=0,
        description="Number of matches from task context collection",
    )

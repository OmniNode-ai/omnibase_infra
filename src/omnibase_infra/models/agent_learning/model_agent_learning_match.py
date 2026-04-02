# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Match result model for agent learning retrieval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.agent_learning.enum_learning_match_type import (
    EnumLearningMatchType,
)
from omnibase_infra.models.agent_learning.model_agent_learning import (
    ModelAgentLearning,
)


class ModelAgentLearningMatch(BaseModel):
    """A single learning match result with relevance scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_type: EnumLearningMatchType = Field(
        ...,
        description="How this match was found (error_signature or task_context)",
    )
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from vector search",
    )
    freshness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Freshness decay factor (1.0 = today, decays 10%/week)",
    )
    combined_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final ranking score: similarity * freshness_score",
    )
    learning: ModelAgentLearning = Field(
        ...,
        description="The matched learning record",
    )

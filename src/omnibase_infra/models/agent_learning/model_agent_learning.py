# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic model for agent learning records.

A learning record captures what an agent discovered during a successful session:
which errors it hit, what files it touched, and how it resolved the problem.
These records are stored in Postgres (metadata) and Qdrant (vectors) for
retrieval by future agents at decision points.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.agent_learning.enum_learning_task_type import (
    EnumLearningTaskType,
)


class ModelAgentLearning(BaseModel):
    """A structured learning record from a successful agent session."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Primary key")
    session_id: UUID = Field(
        ..., description="Source session that produced this learning"
    )
    repo: str = Field(..., min_length=1, max_length=128, description="Repository name")
    file_paths_touched: tuple[str, ...] = Field(
        default=(),
        description="Files the agent edited or created during the session",
    )
    error_signatures: tuple[str, ...] = Field(
        default=(),
        description="Error messages encountered during the session (from failed tool outputs)",
    )
    resolution_summary: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="LLM-generated 2-3 sentence summary of what the agent did to resolve the problem",
    )
    ticket_id: str | None = Field(
        default=None,
        max_length=64,
        description="Ticket ID extracted from git branch name",
    )
    task_type: EnumLearningTaskType = Field(
        default=EnumLearningTaskType.UNKNOWN,
        description="Classified type of work the agent was performing",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the learning quality (based on outcome clarity)",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this learning has been retrieved",
    )
    last_accessed_at: datetime | None = Field(
        default=None,
        description="Last time this learning was retrieved by another agent",
    )
    created_at: datetime = Field(
        ...,
        description="When this learning was extracted (UTC, timezone-aware)",
    )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for PR list effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPRListRequest(BaseModel):
    """Request to list open PRs across GitHub repositories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    repos: tuple[str, ...] = Field(
        ..., description="GitHub repos to scan (e.g., 'OmniNode-ai/omniclaude')."
    )
    authors: tuple[str, ...] = Field(
        default_factory=tuple, description="Filter by PR author usernames."
    )
    labels: tuple[str, ...] = Field(
        default_factory=tuple, description="Filter by PR labels (any match)."
    )
    since: str = Field(
        default="", description="Filter PRs updated after this ISO 8601 date."
    )

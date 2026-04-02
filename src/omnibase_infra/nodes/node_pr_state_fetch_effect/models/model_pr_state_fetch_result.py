# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for PR state fetch effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPrStateFetchResult(BaseModel):
    """Result of fetching a PR's state from GitHub."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    pr_number: int = Field(..., description="GitHub PR number.")
    repo: str = Field(..., description="GitHub repo slug.")
    mergeable: str = Field(default="UNKNOWN", description="GitHub mergeable state.")
    merge_state_status: str = Field(
        default="UNKNOWN", description="GitHub mergeStateStatus."
    )
    review_decision: str = Field(
        default="",
        description="GitHub reviewDecision (APPROVED, CHANGES_REQUESTED, etc.).",
    )
    ci_status: str = Field(
        default="UNKNOWN", description="Overall CI status (SUCCESS, FAILURE, PENDING)."
    )
    additions: int = Field(default=0, description="Lines added.")
    deletions: int = Field(default=0, description="Lines deleted.")
    changed_files: int = Field(default=0, description="Number of changed files.")
    base_ref: str = Field(default="main", description="Base branch name.")
    head_ref: str = Field(default="", description="Head branch name.")
    title: str = Field(default="", description="PR title.")
    success: bool = Field(default=True, description="Whether the fetch succeeded.")
    error_message: str = Field(default="", description="Error message if fetch failed.")

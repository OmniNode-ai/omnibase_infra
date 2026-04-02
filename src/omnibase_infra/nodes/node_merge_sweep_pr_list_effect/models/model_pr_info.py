# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model representing a single PR's metadata for merge-sweep classification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPRInfo(BaseModel):
    """Metadata for a single GitHub pull request."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    number: int = Field(..., description="PR number.")
    repo: str = Field(..., description="Full repo name (org/repo).")
    title: str = Field(default="", description="PR title.")
    head_ref: str = Field(default="", description="Head branch name.")
    base_ref: str = Field(default="main", description="Base branch name.")
    author: str = Field(default="", description="PR author login.")
    is_draft: bool = Field(default=False, description="Whether PR is a draft.")
    mergeable: str = Field(
        default="UNKNOWN",
        description="Mergeable status: MERGEABLE, CONFLICTING, UNKNOWN.",
    )
    review_decision: str = Field(
        default="", description="Review decision: APPROVED, CHANGES_REQUESTED, etc."
    )
    ci_status: str = Field(
        default="UNKNOWN",
        description="Rollup CI status: SUCCESS, FAILURE, PENDING, UNKNOWN.",
    )
    has_auto_merge: bool = Field(
        default=False, description="Whether auto-merge is already enabled."
    )
    labels: tuple[str, ...] = Field(
        default_factory=tuple, description="PR label names."
    )
    updated_at: str = Field(default="", description="ISO 8601 last updated timestamp.")

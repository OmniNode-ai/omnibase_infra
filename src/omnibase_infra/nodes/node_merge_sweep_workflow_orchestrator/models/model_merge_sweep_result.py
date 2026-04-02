# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for merge-sweep workflow."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelMergeSweepResult(BaseModel):
    """Final result of the merge-sweep workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    status: Literal["queued", "nothing_to_merge", "partial", "complete", "error"] = (
        Field(..., description="Overall workflow status.")
    )
    total_prs_scanned: int = Field(default=0, description="Total PRs found.")
    track_a_count: int = Field(default=0, description="PRs classified as Track A.")
    track_b_count: int = Field(default=0, description="PRs classified as Track B.")
    skipped_count: int = Field(default=0, description="PRs skipped (drafts, etc).")
    auto_merge_enabled: int = Field(
        default=0, description="PRs with auto-merge enabled."
    )
    auto_merge_failed: int = Field(
        default=0, description="PRs where auto-merge failed."
    )
    repos_scanned: tuple[str, ...] = Field(
        default_factory=tuple, description="Repos successfully scanned."
    )
    repos_failed: tuple[str, ...] = Field(
        default_factory=tuple, description="Repos that failed to scan."
    )
    error_message: str = Field(default="", description="Error if workflow failed.")
    dry_run: bool = Field(default=False, description="Whether this was a dry run.")

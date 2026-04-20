# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome model for a single PR auto-merge operation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelAutoMergeOutcome(BaseModel):
    """Outcome of enabling auto-merge on a single PR."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo: str = Field(..., description="Full repo name.")
    pr_number: int = Field(..., description="PR number.")
    success: bool = Field(default=True, description="Whether auto-merge was enabled.")
    error_message: str = Field(default="", description="Error if failed.")
    dry_run: bool = Field(default=False, description="Whether this was a dry run.")
    enqueued: bool = Field(
        default=False,
        description=(
            "Whether the PR was successfully enqueued into the merge queue via "
            "the enqueuePullRequest GraphQL mutation. Auto-merge alone is not "
            "sufficient for merge-queue repos — the PR must be explicitly enqueued."
        ),
    )
    queue_position: int | None = Field(
        default=None,
        description="Merge queue position returned by enqueuePullRequest, if enqueued.",
    )
    enqueue_error: str = Field(
        default="",
        description=(
            "Error from enqueuePullRequest attempt. Empty if the call succeeded "
            "or the repo does not use a merge queue (enqueue_skipped=True)."
        ),
    )
    enqueue_skipped: bool = Field(
        default=False,
        description=(
            "True when enqueue was intentionally skipped — e.g. dry_run, the repo "
            "does not have a merge queue configured, or auto-merge itself failed."
        ),
    )

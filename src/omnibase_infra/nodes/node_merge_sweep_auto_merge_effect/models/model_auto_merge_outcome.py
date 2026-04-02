# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome model for a single PR auto-merge operation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelAutoMergeOutcome(BaseModel):
    """Outcome of enabling auto-merge on a single PR."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(..., description="Full repo name.")
    pr_number: int = Field(..., description="PR number.")
    success: bool = Field(default=True, description="Whether auto-merge was enabled.")
    error_message: str = Field(default="", description="Error if failed.")
    dry_run: bool = Field(default=False, description="Whether this was a dry run.")

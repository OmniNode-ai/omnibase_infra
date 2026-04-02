# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for auto-merge effect."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelAutoMergeRequest(BaseModel):
    """Request to enable auto-merge on a set of PRs."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    prs: tuple[tuple[str, int], ...] = Field(
        ..., description="Tuples of (repo, pr_number) to enable auto-merge on."
    )
    merge_method: Literal["squash", "merge", "rebase"] = Field(
        default="squash", description="Merge method to use."
    )
    dry_run: bool = Field(
        default=False, description="If true, report what would happen without acting."
    )

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command model to initiate a merge-sweep workflow."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelMergeSweepCommand(BaseModel):
    """Command payload to trigger merge-sweep workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Unique correlation ID for this workflow run."
    )
    repos: tuple[str, ...] = Field(
        ..., description="GitHub repos to scan (org/repo format)."
    )
    merge_method: Literal["squash", "merge", "rebase"] = Field(
        default="squash", description="Merge strategy."
    )
    require_approval: bool = Field(
        default=True, description="Require review approval for Track A."
    )
    dry_run: bool = Field(
        default=False, description="Report only, do not enable auto-merge."
    )
    authors: tuple[str, ...] = Field(
        default_factory=tuple, description="Filter by PR author."
    )
    labels: tuple[str, ...] = Field(
        default_factory=tuple, description="Filter by PR label."
    )
    since: str = Field(default="", description="ISO 8601 date filter for updatedAt.")

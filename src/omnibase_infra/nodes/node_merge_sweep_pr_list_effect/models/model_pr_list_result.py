# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for PR list effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)


class ModelPRListResult(BaseModel):
    """Result of listing open PRs across repositories."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    prs: tuple[ModelPRInfo, ...] = Field(
        default_factory=tuple, description="All open PRs found."
    )
    repos_scanned: tuple[str, ...] = Field(
        default_factory=tuple, description="Repos that were successfully scanned."
    )
    repos_failed: tuple[str, ...] = Field(
        default_factory=tuple, description="Repos that failed to scan."
    )
    success: bool = Field(default=True, description="Whether the scan succeeded.")
    error_message: str = Field(default="", description="Error message if scan failed.")

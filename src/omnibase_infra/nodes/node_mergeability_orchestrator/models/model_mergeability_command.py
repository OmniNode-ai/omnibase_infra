# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command model to initiate a mergeability-gate workflow."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelMergeabilityCommand(BaseModel):
    """Command payload emitted by bin-shell to trigger mergeability-gate workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ..., description="Unique correlation ID for this workflow run."
    )
    pr_number: int = Field(..., description="GitHub PR number to evaluate.")
    repo: str = Field(..., description="GitHub repo slug (org/repo).")
    linear_ticket_ref: str = Field(
        default="", description="Linear ticket reference for test evidence (optional)."
    )

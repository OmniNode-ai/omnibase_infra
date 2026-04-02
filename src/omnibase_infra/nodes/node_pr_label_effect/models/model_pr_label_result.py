# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for PR label effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPrLabelResult(BaseModel):
    """Result of applying mergeability labels to a GitHub PR."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    pr_number: int = Field(..., description="GitHub PR number.")
    repo: str = Field(..., description="GitHub repo slug.")
    label_applied: str = Field(..., description="Label that was applied.")
    labels_removed: tuple[str, ...] = Field(
        default_factory=tuple, description="Labels that were removed."
    )
    success: bool = Field(
        default=True, description="Whether the label operation succeeded."
    )
    error_message: str = Field(
        default="", description="Error message if labeling failed."
    )

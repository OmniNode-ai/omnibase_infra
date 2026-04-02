# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command model to initiate a scope-check workflow."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScopeCheckCommand(BaseModel):
    """Command payload emitted by bin-shell to trigger scope-check workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Unique correlation ID for this workflow run."
    )
    plan_file_path: str = Field(
        ..., description="Absolute path to the plan file to read."
    )
    output_path: str = Field(
        default="~/.claude/scope-manifest.json",
        description="Path to write the scope manifest JSON.",
    )
    auto_confirm: bool = Field(
        default=False,
        description="Skip interactive confirmation (always true in ONEX workflow).",
    )

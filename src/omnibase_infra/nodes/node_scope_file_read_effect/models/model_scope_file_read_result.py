# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for scope file read effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScopeFileReadResult(BaseModel):
    """Result of reading a plan file from the filesystem."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    file_path: str = Field(..., description="Path that was read.")
    output_path: str = Field(
        default="~/.claude/scope-manifest.json",
        description="Path to write the scope manifest JSON.",
    )
    content: str = Field(..., description="File content.")
    success: bool = Field(default=True, description="Whether the read succeeded.")
    error_message: str = Field(default="", description="Error message if read failed.")

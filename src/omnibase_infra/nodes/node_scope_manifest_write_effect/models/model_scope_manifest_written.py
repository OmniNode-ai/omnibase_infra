# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for scope manifest write effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScopeManifestWritten(BaseModel):
    """Confirmation that the scope manifest was written to disk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    manifest_path: str = Field(
        ..., description="Absolute path where manifest was written."
    )
    success: bool = Field(default=True, description="Whether the write succeeded.")
    error_message: str = Field(default="", description="Error message if write failed.")

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for scope file read effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.local_state_paths import load_local_state_paths


class ModelScopeFileReadRequest(BaseModel):
    """Request to read a plan file from the filesystem."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    file_path: str = Field(..., description="Absolute path to the plan file.")
    output_path: str = Field(
        default_factory=lambda: load_local_state_paths().scope_manifest_path,
        description="Path to write the scope manifest JSON.",
    )

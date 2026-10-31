# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for scope file read effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScopeFileReadRequest(BaseModel):
    """Request to read a plan file from the filesystem."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    file_path: str = Field(..., description="Absolute path to the plan file.")
    output_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to write the scope manifest JSON. Required: the command"
            " producer supplies it and every hop carries it forward; code holds"
            " no default location (OMN-21799)."
        ),
    )

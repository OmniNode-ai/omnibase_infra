# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output model for scope extraction compute node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScopeExtracted(BaseModel):
    """Extracted scope manifest from a plan file."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    plan_file_path: str = Field(..., description="Source plan file path.")
    output_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to write the scope manifest JSON. Required: the command"
            " producer supplies it and every hop carries it forward; code holds"
            " no default location (OMN-21799)."
        ),
    )
    files: tuple[str, ...] = Field(default_factory=tuple, description="Files in scope.")
    directories: tuple[str, ...] = Field(
        default_factory=tuple, description="Directories in scope."
    )
    repos: tuple[str, ...] = Field(default_factory=tuple, description="Repos in scope.")
    systems: tuple[str, ...] = Field(
        default_factory=tuple, description="Systems in scope."
    )
    adjacent_files: tuple[str, ...] = Field(
        default_factory=tuple, description="Adjacent files that may need modification."
    )

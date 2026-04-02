# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for scope extraction compute node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScopeExtractInput(BaseModel):
    """Input containing plan file content for scope extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    plan_file_path: str = Field(..., description="Original path of the plan file.")
    content: str = Field(..., description="Plan file content to parse.")

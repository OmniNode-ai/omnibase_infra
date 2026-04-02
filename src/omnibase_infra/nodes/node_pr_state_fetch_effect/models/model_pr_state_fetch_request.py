# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for PR state fetch effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPrStateFetchRequest(BaseModel):
    """Request to fetch a PR's state from GitHub."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    pr_number: int = Field(..., description="GitHub PR number.")
    repo: str = Field(..., description="GitHub repo slug (org/repo).")

# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A single model endpoint to probe."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelHealthProbeTarget(BaseModel):
    """A single model endpoint to probe."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model_key: str = Field(..., description="Model identifier from registry.")
    base_url: str = Field(..., description="Resolved base URL for health check.")
    transport: str = Field(default="http", description="Transport type: http or sdk.")
